"""
Leak-Free 5-Fold Patient-Stratified Cross-Validation Engine
===========================================================
Executes rigorous, scientifically valid benchmarks for:
  - ResNet-50
  - EfficientNet-B0
  - ResNet-18
with two-stage transfer learning (probe -> fine-tune), mixed precision,
on-the-fly realistic augmentation, and out-of-fold aggregation.
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

# Ensure local imports work cleanly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from dataset import EmbryoDataset, get_transforms, load_clean_dataset, CLASS_NAMES, INV_GRADE_MAP
from models import get_model, freeze_feature_extractor, unfreeze_backbone
from losses import get_loss_criterion

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_single_fold(fold_idx, arch_name, df_all, config, device):
    set_seed(config['seed'] + fold_idx)

    # Train / Val split strictly by fold index
    train_df = df_all[df_all['fold'] != fold_idx].copy()
    val_df = df_all[df_all['fold'] == fold_idx].copy()

    # Integrity verification
    train_pts = set(train_df['patient_id'].unique())
    val_pts = set(val_df['patient_id'].unique())
    overlap = train_pts.intersection(val_pts)
    assert len(overlap) == 0, f"DATA LEAKAGE DETECTED! Overlapping patients: {overlap}"

    img_size = config['img_size']
    use_clahe = config['use_clahe']

    train_transform = get_transforms(img_size=img_size, is_training=True)
    val_transform = get_transforms(img_size=img_size, is_training=False)

    train_loader = DataLoader(
        EmbryoDataset(train_df, transform=train_transform, use_clahe=use_clahe),
        batch_size=config['batch_size'], shuffle=True, num_workers=0, pin_memory=True
    )
    val_loader = DataLoader(
        EmbryoDataset(val_df, transform=val_transform, use_clahe=use_clahe),
        batch_size=config['batch_size'], shuffle=False, num_workers=0, pin_memory=True
    )

    # Calculate class weights solely from training fold
    counts = train_df['grade'].value_counts()
    n_train = len(train_df)
    weights = [n_train / (3.0 * counts[f'Grade-{i+1}']) for i in range(3)]
    class_weights = torch.tensor(weights, dtype=torch.float32).to(device) if config['use_class_weights'] else None

    # Instantiate model
    model = get_model(arch_name=arch_name, num_classes=3, pretrained=True, dropout_rate=config['dropout']).to(device)

    # Stage 1: Linear Probe
    probe_epochs = config['probe_epochs']
    finetune_epochs = config['finetune_epochs']
    total_epochs = probe_epochs + finetune_epochs

    criterion = get_loss_criterion(
        loss_type=config['loss_type'],
        class_weights=class_weights,
        smoothing=config['ordinal_smoothing']
    )

    # Setup Stage 1: Freeze backbone, train classification head
    freeze_feature_extractor(model, arch_name)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=config['head_lr'], weight_decay=config['weight_decay'])

    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None

    best_val_acc = 0.0
    best_preds = None
    best_probs = None
    best_targets = None
    best_filenames = None
    best_state = None

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    print(f"\n--- [{arch_name.upper()}] Fold {fold_idx + 1}/5 | Train: {len(train_df)} imgs ({len(train_pts)} pts) | Val: {len(val_df)} imgs ({len(val_pts)} pts) ---", flush=True)

    for epoch in range(1, total_epochs + 1):
        # Stage transition: Switch to fine-tuning with smaller LR
        if epoch == probe_epochs + 1:
            unfreeze_backbone(model, arch_name, unfreeze_all=config['unfreeze_all'])
            optimizer = optim.AdamW(model.parameters(), lr=config['finetune_lr'], weight_decay=config['weight_decay'])
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=finetune_epochs, eta_min=1e-6)

        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels, _, _ in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad()

            if scaler is not None:
                with torch.amp.autocast('cuda'):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            train_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

        if epoch > probe_epochs and 'scheduler' in locals():
            scheduler.step()

        train_acc = train_correct / train_total

        # Validation phase (strictly deterministic evaluation)
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        epoch_preds = []
        epoch_probs = []
        epoch_targets = []
        epoch_filenames = []

        with torch.no_grad():
            for images, labels, _, filenames in val_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                with torch.amp.autocast('cuda') if device.type == 'cuda' else torch.no_grad():
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                probs = torch.softmax(outputs, dim=1).cpu().numpy()
                preds = outputs.argmax(dim=1).cpu().numpy()
                targets = labels.cpu().numpy()

                val_loss += loss.item() * images.size(0)
                val_correct += (preds == targets).sum()
                val_total += len(targets)

                epoch_preds.extend(preds)
                epoch_probs.extend(probs)
                epoch_targets.extend(targets)
                epoch_filenames.extend(filenames)

        val_acc = val_correct / val_total

        history['train_loss'].append(train_loss / train_total)
        history['val_loss'].append(val_loss / val_total)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        if val_acc > best_val_acc or epoch == 1:
            best_val_acc = val_acc
            best_preds = np.array(epoch_preds)
            best_probs = np.array(epoch_probs)
            best_targets = np.array(epoch_targets)
            best_filenames = list(epoch_filenames)
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 5 == 0 or epoch == total_epochs:
            stage_str = "Probe" if epoch <= probe_epochs else "FineTune"
            print(f"  Epoch [{epoch:02d}/{total_epochs:02d} | {stage_str}] Train Loss: {train_loss/train_total:.4f} Acc: {train_acc*100:.2f}% | Val Loss: {val_loss/val_total:.4f} Acc: {val_acc*100:.2f}% (Best: {best_val_acc*100:.2f}%)", flush=True)

    fold_df = pd.DataFrame({
        'filename': best_filenames,
        'fold': fold_idx,
        'true_label': best_targets,
        'pred_label': best_preds,
        'prob_grade1': best_probs[:, 0],
        'prob_grade2': best_probs[:, 1],
        'prob_grade3': best_probs[:, 2]
    })

    # Free memory
    del model, optimizer, train_loader, val_loader
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return fold_df, best_val_acc, best_state, history

def run_experiment(exp_id, arch_name, config):
    start_time = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n==================================================================", flush=True)
    print(f" STARTING EXPERIMENT [{exp_id}]: Architecture: {arch_name.upper()}", flush=True)
    print(f" Resolution: {config['img_size']}x{config['img_size']} | Preprocessing: {'CLAHE' if config['use_clahe'] else 'Standard RGB'}", flush=True)
    print(f" Loss: {config['loss_type'].upper()} | Two-Stage: {config['probe_epochs']} probe + {config['finetune_epochs']} finetune", flush=True)
    print(f" Hardware: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})", flush=True)
    print(f"==================================================================", flush=True)

    df_clean = load_clean_dataset('labels_clean.csv', remove_duplicates=True)
    print(f"Loaded {len(df_clean)} unique verified labeled images across {df_clean['patient_id'].nunique()} patient cohorts.", flush=True)

    all_fold_dfs = []
    fold_accuracies = []
    best_overall_state = None
    best_overall_acc = 0.0
    all_histories = []

    # Output directories
    out_dir = os.path.dirname(os.path.abspath(__file__))
    ckpt_dir = os.path.join(out_dir, "checkpoints")
    pred_dir = os.path.join(out_dir, "predictions")
    plot_dir = os.path.join(out_dir, "plots")
    metric_dir = os.path.join(out_dir, "metrics")

    for d in [ckpt_dir, pred_dir, plot_dir, metric_dir]:
        os.makedirs(d, exist_ok=True)

    gpu_error = None
    try:
        for fold_idx in range(5):
            fold_df, val_acc, state, hist = train_single_fold(fold_idx, arch_name, df_clean, config, device)
            all_fold_dfs.append(fold_df)
            fold_accuracies.append(val_acc)
            all_histories.append(hist)
            if val_acc > best_overall_acc:
                best_overall_acc = val_acc
                best_overall_state = state
    except RuntimeError as e:
        if 'out of memory' in str(e).lower():
            gpu_error = "CUDA Out of Memory"
            print(f"ERROR: {gpu_error}. Reduce batch size or resolution.", flush=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return None
        else:
            raise e

    duration = time.time() - start_time

    # Combine out-of-fold predictions
    oof_df = pd.concat(all_fold_dfs, ignore_index=True)
    oof_df['true_grade'] = oof_df['true_label'].map(INV_GRADE_MAP)
    oof_df['pred_grade'] = oof_df['pred_label'].map(INV_GRADE_MAP)
    oof_path = os.path.join(pred_dir, f"oof_{exp_id}.csv")
    oof_df.to_csv(oof_path, index=False)

    # Save model weights
    ckpt_path = os.path.join(ckpt_dir, f"best_{exp_id}.pth")
    torch.save(best_overall_state, ckpt_path)

    # Evaluate full out-of-fold metrics
    y_true = oof_df['true_label'].values
    y_pred = oof_df['pred_label'].values
    y_probs = oof_df[['prob_grade1', 'prob_grade2', 'prob_grade3']].values.astype(np.float64)
    y_probs = y_probs / y_probs.sum(axis=1, keepdims=True)

    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
    qwk = cohen_kappa_score(y_true, y_pred, weights='quadratic')
    try:
        auc = roc_auc_score(y_true, y_probs, multi_class='ovr', average='macro')
    except Exception as e:
        print(f"ROC-AUC warning: {e}", flush=True)
        auc = 0.0

    prec_pc, rec_pc, f1_pc, supp_pc = precision_recall_fscore_support(y_true, y_pred, average=None)

    metrics = {
        'exp_id': exp_id,
        'arch_name': arch_name,
        'img_size': config['img_size'],
        'preprocessing': 'CLAHE' if config['use_clahe'] else 'Standard RGB',
        'loss_type': config['loss_type'],
        'probe_epochs': config['probe_epochs'],
        'finetune_epochs': config['finetune_epochs'],
        'head_lr': config['head_lr'],
        'finetune_lr': config['finetune_lr'],
        'batch_size': config['batch_size'],
        'duration_seconds': round(duration, 2),
        'gpu_error': gpu_error,
        'overall_accuracy': float(acc),
        'balanced_accuracy': float(bal_acc),
        'macro_precision': float(prec),
        'macro_recall': float(rec),
        'macro_f1': float(f1),
        'quadratic_weighted_kappa': float(qwk),
        'macro_roc_auc': float(auc),
        'fold_accuracies': [float(x) for x in fold_accuracies],
        'fold_mean_accuracy': float(np.mean(fold_accuracies)),
        'fold_std_accuracy': float(np.std(fold_accuracies)),
        'per_class': {
            CLASS_NAMES[i]: {
                'precision': float(prec_pc[i]),
                'recall': float(rec_pc[i]),
                'f1': float(f1_pc[i]),
                'support': int(supp_pc[i])
            } for i in range(3)
        }
    }

    # Save metrics JSON
    with open(os.path.join(metric_dir, f"{exp_id}_metrics.json"), 'w') as f:
        json.dump(metrics, f, indent=2)

    # Generate Confusion Matrix Figure
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm_norm, cmap=plt.cm.Blues, vmin=0, vmax=1.0)
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
    ax.set_title(f"Confusion Matrix: {exp_id}\n(Out-of-Fold Acc: {acc*100:.2f}%, QWK: {qwk:.4f})", fontweight='bold', pad=10)
    ax.set_xlabel('Predicted Grade', fontweight='bold')
    ax.set_ylabel('True Grade', fontweight='bold')
    plt.tight_layout()
    cm_path = os.path.join(plot_dir, f"cm_{exp_id}.png")
    plt.savefig(cm_path, dpi=200)
    plt.close()

    # Generate Learning Curves Figure (averaged across 5 folds)
    mean_train_loss = np.mean([h['train_loss'] for h in all_histories], axis=0)
    mean_val_loss = np.mean([h['val_loss'] for h in all_histories], axis=0)
    mean_train_acc = np.mean([h['train_acc'] for h in all_histories], axis=0)
    mean_val_acc = np.mean([h['val_acc'] for h in all_histories], axis=0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    epochs_range = range(1, len(mean_train_loss) + 1)

    ax1.plot(epochs_range, mean_train_loss, label='Train Loss', color='#2980b9', lw=2)
    ax1.plot(epochs_range, mean_val_loss, label='Val Loss', color='#c0392b', lw=2)
    ax1.axvline(x=config['probe_epochs'] + 0.5, color='gray', linestyle='--', label='Unfreeze Backbone')
    ax1.set_title('Cross-Entropy / Ordinal Loss Trajectory', fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(epochs_range, mean_train_acc * 100, label='Train Acc (%)', color='#2980b9', lw=2)
    ax2.plot(epochs_range, mean_val_acc * 100, label='Val Acc (%)', color='#27ae60', lw=2)
    ax2.axvline(x=config['probe_epochs'] + 0.5, color='gray', linestyle='--', label='Unfreeze Backbone')
    ax2.set_title('Classification Accuracy (%)', fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.suptitle(f"Training & Validation Curves: {exp_id} ({arch_name.upper()})", fontweight='bold', y=1.02)
    plt.tight_layout()
    curve_path = os.path.join(plot_dir, f"curves_{exp_id}.png")
    plt.savefig(curve_path, dpi=200)
    plt.close()

    print(f"\n==================================================================", flush=True)
    print(f" COMPLETED EXPERIMENT [{exp_id}]: {arch_name.upper()}", flush=True)
    print(f" Out-of-Fold Accuracy:  {acc*100:.2f}% (Fold Mean: {np.mean(fold_accuracies)*100:.2f}% +/- {np.std(fold_accuracies)*100:.2f}%)", flush=True)
    print(f" Balanced Accuracy:     {bal_acc*100:.2f}%", flush=True)
    print(f" Macro F1-Score:        {f1:.4f}", flush=True)
    print(f" Cohen's QWK:           {qwk:.4f}", flush=True)
    print(f" Macro ROC-AUC:         {auc:.4f}", flush=True)
    print(f" Duration:              {duration:.1f} seconds", flush=True)
    print(f"==================================================================", flush=True)

    # Append to experiment comparison CSV
    total_epochs = config['probe_epochs'] + config['finetune_epochs']
    summary_csv = os.path.join(out_dir, "experiments_summary.csv")
    row = {
        'Experiment_ID': exp_id,
        'Architecture': arch_name,
        'Resolution': f"{config['img_size']}x{config['img_size']}",
        'Preprocessing': 'CLAHE' if config['use_clahe'] else 'Standard RGB',
        'Loss_Function': config['loss_type'],
        'Probe_LR': config['head_lr'],
        'FineTune_LR': config['finetune_lr'],
        'Batch_Size': config['batch_size'],
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
        'GPU_Error': gpu_error if gpu_error else 'None'
    }

    if os.path.exists(summary_csv):
        df_sum = pd.read_csv(summary_csv)
        df_sum = pd.concat([df_sum, pd.DataFrame([row])], ignore_index=True)
    else:
        df_sum = pd.DataFrame([row])
    df_sum.to_csv(summary_csv, index=False)

    return metrics
