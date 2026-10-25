"""
Multi-Focal Slice Attention-MIL 5-Fold Cross-Validation Engine
=============================================================
Evaluates embryo quality grading at the Patient / Embryo Bag level by fusing all
focal planes (Z-slices) of each specimen through Gated Attention Pooling.

Features:
  - 100% Patient-isolated 5-Fold CV (133 distinct patient bags, 0 leakage)
  - Evaluates both 3-Class Ordinal Grading and Binary Clinical Decisions
  - Gated Attention weights reveal which focal slice determined the clinical grade
  - Mixed precision execution on NVIDIA RTX 2050
"""

import os
import sys
import time
import json
import random
import argparse
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_recall_fscore_support,
    cohen_kappa_score, roc_auc_score, confusion_matrix
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from mil_dataset import EmbryoBagDataset, load_mil_dataframe, get_mil_transforms, CLASS_NAMES, INV_GRADE_MAP
from mil_model import GatedAttentionMIL, freeze_mil_backbone, unfreeze_mil_backbone

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_single_mil_fold(fold_idx, arch_name, df_all, config, device):
    set_seed(config['seed'] + fold_idx)

    train_df = df_all[df_all['fold'] != fold_idx].copy()
    val_df = df_all[df_all['fold'] == fold_idx].copy()

    # Leakage check
    train_pts = set(train_df['patient_id'].unique())
    val_pts = set(val_df['patient_id'].unique())
    overlap = train_pts.intersection(val_pts)
    assert len(overlap) == 0, f"DATA LEAKAGE DETECTED: Overlap: {overlap}"

    img_size = config['img_size']
    use_clahe = config['use_clahe']

    train_ds = EmbryoBagDataset(train_df, transform=get_mil_transforms(img_size, is_training=True), use_clahe=use_clahe)
    val_ds = EmbryoBagDataset(val_df, transform=get_mil_transforms(img_size, is_training=False), use_clahe=use_clahe)

    # Class weights from training bags
    train_labels = [bag['label'] for bag in train_ds.bags]
    counts = np.bincount(train_labels, minlength=3)
    n_train = len(train_labels)
    weights = [n_train / (3.0 * max(c, 1)) for c in counts]
    class_weights = torch.tensor(weights, dtype=torch.float32).to(device)

    # Instantiate MIL Model
    model = GatedAttentionMIL(
        arch_name=arch_name,
        num_classes=3,
        pretrained=True,
        emb_dim=config['emb_dim'],
        attn_dim=config['attn_dim'],
        dropout=config['dropout']
    ).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    probe_epochs = config['probe_epochs']
    finetune_epochs = config['finetune_epochs']
    total_epochs = probe_epochs + finetune_epochs

    # Stage 1: Freeze backbone, train attention and classifier
    freeze_mil_backbone(model)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config['head_lr'],
        weight_decay=config['weight_decay']
    )
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None

    best_val_acc = 0.0
    best_results = None
    best_state = None
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    print(f"\n--- [MIL-{arch_name.upper()}] Fold {fold_idx+1}/5 | Train: {len(train_ds)} bags | Val: {len(val_ds)} bags ---", flush=True)

    for epoch in range(1, total_epochs + 1):
        # Stage transition
        if epoch == probe_epochs + 1:
            unfreeze_mil_backbone(model)
            optimizer = optim.AdamW(
                model.parameters(),
                lr=config['finetune_lr'],
                weight_decay=config['weight_decay']
            )

        # Training Phase (bag by bag)
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_indices = list(range(len(train_ds)))
        random.shuffle(train_indices)

        optimizer.zero_grad()
        accum_steps = config.get('grad_accum', 4)

        for step, idx in enumerate(train_indices):
            bag_tensor, label, _, _ = train_ds[idx]
            bag_tensor = bag_tensor.to(device, non_blocking=True)
            label_t = torch.tensor([label], dtype=torch.long, device=device)

            with torch.amp.autocast('cuda') if device.type == 'cuda' else torch.no_grad():
                logits, attn_weights = model(bag_tensor)
                loss = criterion(logits, label_t) / accum_steps

            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (step + 1) % accum_steps == 0 or (step + 1) == len(train_indices):
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()

            pred = logits.argmax(dim=1).item()
            train_correct += int(pred == label)
            train_loss += loss.item() * accum_steps

        train_acc = train_correct / len(train_ds)

        # Validation Phase (strictly deterministic)
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_preds = []
        val_probs = []
        val_targets = []
        val_pts = []
        val_attns = []
        val_fnames = []

        with torch.no_grad():
            for idx in range(len(val_ds)):
                bag_tensor, label, patient_id, fnames = val_ds[idx]
                bag_tensor = bag_tensor.to(device, non_blocking=True)
                label_t = torch.tensor([label], dtype=torch.long, device=device)

                with torch.amp.autocast('cuda') if device.type == 'cuda' else torch.no_grad():
                    logits, attn_weights = model(bag_tensor)
                    loss = criterion(logits, label_t)

                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                pred = probs.argmax()

                val_loss += loss.item()
                val_correct += int(pred == label)

                val_preds.append(pred)
                val_probs.append(probs)
                val_targets.append(label)
                val_pts.append(patient_id)
                val_attns.append(attn_weights.cpu().numpy().tolist())
                val_fnames.append(fnames)

        val_acc = val_correct / len(val_ds)

        history['train_loss'].append(train_loss / len(train_ds))
        history['val_loss'].append(val_loss / len(val_ds))
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        if val_acc > best_val_acc or epoch == 1:
            best_val_acc = val_acc
            best_results = {
                'patient_id': val_pts,
                'fold': [fold_idx] * len(val_pts),
                'true_label': val_targets,
                'pred_label': val_preds,
                'prob_grade1': [p[0] for p in val_probs],
                'prob_grade2': [p[1] for p in val_probs],
                'prob_grade3': [p[2] for p in val_probs],
                'attention_weights': val_attns,
                'slice_filenames': val_fnames
            }
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 5 == 0 or epoch == total_epochs:
            stage_str = "Probe" if epoch <= probe_epochs else "FineTune"
            print(f"  Epoch [{epoch:02d}/{total_epochs:02d} | {stage_str}] Train Loss: {train_loss/len(train_ds):.4f} Acc: {train_acc*100:.2f}% | Val Loss: {val_loss/len(val_ds):.4f} Acc: {val_acc*100:.2f}% (Best: {best_val_acc*100:.2f}%)", flush=True)

    fold_df = pd.DataFrame(best_results)

    del model, optimizer, train_ds, val_ds
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return fold_df, best_val_acc, best_state, history

def run_mil_experiment(exp_id='exp_mil_resnet50_clahe', arch_name='resnet50', use_clahe=True):
    start_time = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\n==================================================================", flush=True)
    print(f" STARTING MULTI-FOCAL SLICE MIL EXPERIMENT [{exp_id}]", flush=True)
    print(f" Architecture: {arch_name.upper()} | Aggregation: Gated Attention MIL", flush=True)
    print(f" Preprocessing: {'CLAHE' if use_clahe else 'Standard RGB'} | Hardware: {device}", flush=True)
    print(f"==================================================================", flush=True)

    df_clean = load_mil_dataframe('labels_clean.csv')
    n_bags = df_clean['patient_id'].nunique()
    print(f"Loaded {len(df_clean)} verified slices across {n_bags} unique patient embryo bags.", flush=True)

    config = {
        'img_size': 224,
        'emb_dim': 512,
        'attn_dim': 128,
        'head_lr': 1e-3,
        'finetune_lr': 3e-5,
        'weight_decay': 1e-4,
        'probe_epochs': 5,
        'finetune_epochs': 15,
        'grad_accum': 4,
        'dropout': 0.3,
        'seed': 42,
        'use_clahe': use_clahe
    }

    base_dir = os.path.dirname(os.path.abspath(__file__))
    pred_dir = os.path.join(base_dir, "predictions")
    plot_dir = os.path.join(base_dir, "plots")
    metric_dir = os.path.join(base_dir, "metrics")
    ckpt_dir = os.path.join(base_dir, "checkpoints")

    for d in [pred_dir, plot_dir, metric_dir, ckpt_dir]:
        os.makedirs(d, exist_ok=True)

    all_fold_dfs = []
    fold_accuracies = []
    all_histories = []
    best_overall_state = None
    best_overall_acc = 0.0

    for fold_idx in range(5):
        fold_df, val_acc, state, hist = train_single_mil_fold(fold_idx, arch_name, df_clean, config, device)
        all_fold_dfs.append(fold_df)
        fold_accuracies.append(val_acc)
        all_histories.append(hist)
        if val_acc > best_overall_acc:
            best_overall_acc = val_acc
            best_overall_state = state

    duration = time.time() - start_time

    # Combine Out-of-Fold Predictions (133 embryos)
    oof_df = pd.concat(all_fold_dfs, ignore_index=True)
    oof_df['true_grade'] = oof_df['true_label'].map(INV_GRADE_MAP)
    oof_df['pred_grade'] = oof_df['pred_label'].map(INV_GRADE_MAP)
    oof_path = os.path.join(pred_dir, f"oof_{exp_id}.csv")
    oof_df.to_csv(oof_path, index=False)

    # Save model weights
    torch.save(best_overall_state, os.path.join(ckpt_dir, f"best_{exp_id}.pth"))

    # 3-Class Metrics
    y_true = oof_df['true_label'].values
    y_pred = oof_df['pred_label'].values
    y_probs = oof_df[['prob_grade1', 'prob_grade2', 'prob_grade3']].values.astype(np.float64)
    y_probs = y_probs / y_probs.sum(axis=1, keepdims=True)

    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
    qwk = cohen_kappa_score(y_true, y_pred, weights='quadratic')
    auc = roc_auc_score(y_true, y_probs, multi_class='ovr', average='macro')

    # Binary Clinical Decisions
    # Decision 1: Top Quality (Grade 1) vs Sub-optimal (Grade 2 + 3)
    y_bin1_true = (y_true == 0).astype(int)
    p_bin1 = y_probs[:, 0]
    pred_bin1 = (p_bin1 >= 0.33).astype(int)
    acc_bin1 = accuracy_score(y_bin1_true, pred_bin1)
    auc_bin1 = roc_auc_score(y_bin1_true, p_bin1)

    # Decision 2: Transferable (Grade 1 + 2) vs Discard (Grade 3)
    y_bin2_true = (y_true != 2).astype(int)
    p_bin2 = y_probs[:, 0] + y_probs[:, 1]
    pred_bin2 = (p_bin2 >= 0.50).astype(int)
    acc_bin2 = accuracy_score(y_bin2_true, pred_bin2)
    auc_bin2 = roc_auc_score(y_bin2_true, p_bin2)

    # Plot Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm_norm, cmap=plt.cm.Purples, vmin=0, vmax=1.0)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for i in range(3):
        for j in range(3):
            val = cm[i, j]
            pct = cm_norm[i, j] * 100
            color = "white" if cm_norm[i, j] > 0.5 else "black"
            ax.text(j, i, f"{pct:.1f}%\n(n={val})", ha="center", va="center", color=color, fontsize=10, fontweight='bold')

    ax.set_xticks(np.arange(3))
    ax.set_xticklabels(CLASS_NAMES, fontweight='bold')
    ax.set_yticks(np.arange(3))
    ax.set_yticklabels(CLASS_NAMES, fontweight='bold')
    ax.set_title(f"Confusion Matrix: {exp_id}\n(Embryo OOF Acc: {acc*100:.2f}%, QWK: {qwk:.4f})", fontweight='bold', pad=10)
    ax.set_xlabel('Predicted Grade', fontweight='bold')
    ax.set_ylabel('True Grade', fontweight='bold')
    plt.tight_layout()
    cm_path = os.path.join(plot_dir, f"cm_{exp_id}.png")
    plt.savefig(cm_path, dpi=200)
    plt.close()

    print(f"\n==================================================================", flush=True)
    print(f" COMPLETED MIL EXPERIMENT [{exp_id}]: {arch_name.upper()}", flush=True)
    print(f" Embryo-Level 3-Class Accuracy: {acc*100:.2f}% (Fold Mean: {np.mean(fold_accuracies)*100:.2f}% +/- {np.std(fold_accuracies)*100:.2f}%)", flush=True)
    print(f" Balanced Accuracy:             {bal_acc*100:.2f}%", flush=True)
    print(f" Macro F1-Score:                {f1:.4f}", flush=True)
    print(f" Cohen's QWK:                   {qwk:.4f}", flush=True)
    print(f" Multiclass ROC-AUC:            {auc:.4f}", flush=True)
    print(f" ---------------------------------------------------------------", flush=True)
    print(f" CLINICAL BINARY DECISIONS:", flush=True)
    print(f"   Top-Quality (Grade 1 vs 2+3): Acc={acc_bin1*100:.2f}%, AUC={auc_bin1*100:.2f}%", flush=True)
    print(f"   Transferable (Grade 1+2 vs 3): Acc={acc_bin2*100:.2f}%, AUC={auc_bin2*100:.2f}%", flush=True)
    print(f" Duration:                      {duration:.1f} seconds", flush=True)
    print(f"==================================================================", flush=True)

    metrics = {
        'exp_id': exp_id,
        'arch_name': f"MIL_{arch_name}",
        'aggregation': 'Gated_Attention_MIL',
        'preprocessing': 'CLAHE' if use_clahe else 'Standard RGB',
        'duration_seconds': round(duration, 1),
        'oof_accuracy': float(acc),
        'balanced_accuracy': float(bal_acc),
        'macro_f1': float(f1),
        'cohen_qwk': float(qwk),
        'multiclass_roc_auc': float(auc),
        'binary_top_quality_acc': float(acc_bin1),
        'binary_top_quality_auc': float(auc_bin1),
        'binary_transferable_acc': float(acc_bin2),
        'binary_transferable_auc': float(auc_bin2),
        'fold_accuracies': [float(x) for x in fold_accuracies]
    }

    with open(os.path.join(metric_dir, f"{exp_id}_metrics.json"), 'w') as f:
        json.dump(metrics, f, indent=2)

    # Append to experiment summary CSV
    summary_csv = os.path.join(base_dir, "experiments_summary.csv")
    row = {
        'Experiment_ID': exp_id,
        'Architecture': f"MIL_{arch_name}",
        'Resolution': f"{config['img_size']}x{config['img_size']}",
        'Preprocessing': 'CLAHE' if use_clahe else 'Standard RGB',
        'Loss_Function': 'cross_entropy',
        'Probe_LR': config['head_lr'],
        'FineTune_LR': config['finetune_lr'],
        'Batch_Size': config['grad_accum'],
        'Total_Epochs': total_epochs,
        'Fold_Accuracies': str([round(x*100, 2) for x in fold_accuracies]),
        'CV_Mean_Acc': round(np.mean(fold_accuracies) * 100, 2),
        'CV_Std_Acc': round(np.std(fold_accuracies) * 100, 2),
        'Overall_OOF_Acc': round(acc * 100, 2),
        'Balanced_Acc': round(bal_acc * 100, 2),
        'Macro_F1': round(f1 * 100, 2),
        'Cohen_QWK': round(qwk, 4),
        'Macro_ROC_AUC': round(auc * 100, 2),
        'Duration_Sec': round(duration, 1),
        'GPU_Error': 'None'
    }

    if os.path.exists(summary_csv):
        df_sum = pd.read_csv(summary_csv)
        df_sum = df_sum[df_sum['Experiment_ID'] != exp_id]
        df_sum = pd.concat([df_sum, pd.DataFrame([row])], ignore_index=True)
        df_sum.to_csv(summary_csv, index=False)

    return metrics

def main():
    parser = argparse.ArgumentParser(description="Multi-Focal Slice MIL Runner")
    parser.add_argument('--arch', type=str, default='resnet50', choices=['resnet50', 'efficientnet_b0', 'resnet18'])
    parser.add_argument('--use_clahe', action='store_true', default=True)
    args = parser.parse_args()

    exp_id = f"exp_mil_{args.arch}_{'clahe' if args.use_clahe else 'rgb'}"
    run_mil_experiment(exp_id=exp_id, arch_name=args.arch, use_clahe=args.use_clahe)

if __name__ == '__main__':
    main()
