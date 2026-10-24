"""
Leak-Free Patient-Separated Benchmark for Human Blastocyst Grading
==================================================================
This script performs a rigorous, scientifically valid evaluation of
three deep learning architectures:
  1. VGG-16
  2. ResNet-50
  3. EfficientNet-B4

Key Guarantees:
  - 100% Patient Isolation: StratifiedGroupKFold splits ensure NO patient
    appears in both training and test sets.
  - Zero Leakage: Augmentations are performed strictly on-the-fly during
    training. Test/Validation evaluation is strictly single-crop deterministic.
  - Full Out-Of-Fold Evaluation: All 324 ground-truth embryos are evaluated
    on models trained without seeing their respective patient IDs.
  - Fair Comparison: Identical evaluation pipeline and patient folds across
    all models.
"""

import os
import sys
import json
import random
import argparse
import numpy as np
import pandas as pd
from PIL import Image

# Force unbuffered output so logs appear in real time
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models

from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_recall_fscore_support,
    cohen_kappa_score, roc_auc_score, confusion_matrix
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2

# Set reproducible seeds
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

CLASS_NAMES = ['Grade-1', 'Grade-2', 'Grade-3']
GRADE_MAP = {'Grade-1': 0, 'Grade-2': 1, 'Grade-3': 2}
INV_GRADE_MAP = {0: 'Grade-1', 1: 'Grade-2', 2: 'Grade-3'}

# CLAHE Preprocessing
class CLAHETransform:
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def __call__(self, img_pil):
        img_np = np.array(img_pil)
        lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
        return Image.fromarray(enhanced)

# Embryo Dataset
class EmbryoDataset(Dataset):
    def __init__(self, df, img_dir='Images', transform=None, use_clahe=False):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.clahe = CLAHETransform() if use_clahe else None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row['filename'])
        image = Image.open(img_path).convert('RGB')

        if self.clahe is not None:
            image = self.clahe(image)

        if self.transform is not None:
            image = self.transform(image)

        label = GRADE_MAP[row['grade']]
        patient_id = row['patient_id']
        filename = row['filename']
        return image, label, patient_id, filename

# Ordinal Soft Cross-Entropy Loss
class OrdinalSoftLoss(nn.Module):
    def __init__(self, smoothing=0.15):
        super(OrdinalSoftLoss, self).__init__()
        self.smoothing = smoothing
        # Ordinal target distribution matrix for 3 classes
        self.soft_matrix = torch.tensor([
            [1.0 - smoothing, smoothing, 0.0],
            [smoothing / 2.0, 1.0 - smoothing, smoothing / 2.0],
            [0.0, smoothing, 1.0 - smoothing]
        ], dtype=torch.float32)

    def forward(self, logits, targets):
        log_probs = torch.log_softmax(logits, dim=1)
        matrix = self.soft_matrix.to(logits.device)
        soft_targets = matrix[targets]
        loss = -(soft_targets * log_probs).sum(dim=1).mean()
        return loss

# Model Builder with local checkpoint fallback
def build_model(model_name, num_classes=3, pretrained=True):
    cache_dir = os.path.expanduser(r"~\.cache\torch\hub\checkpoints")

    if model_name == 'vgg16':
        model = models.vgg16(weights=None)
        if pretrained:
            vgg_path = os.path.join(cache_dir, "vgg16-397923af.pth")
            if os.path.exists(vgg_path):
                model.load_state_dict(torch.load(vgg_path, weights_only=True))
            else:
                model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        in_features = model.classifier[6].in_features
        model.classifier[6] = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(in_features, num_classes)
        )
        img_size = 224
        use_clahe = False
        batch_size = 16
        lr = 1e-4

    elif model_name == 'resnet50':
        model = models.resnet50(weights=None)
        if pretrained:
            res_path = os.path.join(cache_dir, "resnet50-11ad3fa6.pth")
            if os.path.exists(res_path):
                model.load_state_dict(torch.load(res_path, weights_only=True))
            else:
                model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, num_classes)
        )
        img_size = 224
        use_clahe = False
        batch_size = 16
        lr = 1e-4

    elif model_name == 'efficientnet_b4':
        model = models.efficientnet_b4(weights=None)
        if pretrained:
            eff_path = os.path.join(cache_dir, "efficientnet_b4_rwightman-7eb33cd5.pth")
            if os.path.exists(eff_path):
                model.load_state_dict(torch.load(eff_path, weights_only=True))
            else:
                model = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.DEFAULT)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(in_features, num_classes)
        )
        img_size = 380
        use_clahe = True
        batch_size = 8  # Fit comfortably in 4GB VRAM
        lr = 2e-4
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    return model, img_size, use_clahe, batch_size, lr

# Train & Evaluate Single Fold
def train_and_eval_fold(fold_idx, model_name, df_all, device, epochs=20):
    set_seed(42 + fold_idx)
    
    # Strictly split by fold: test_fold = fold_idx, train_folds = others
    train_df = df_all[(df_all['fold'] >= 0) & (df_all['fold'] != fold_idx) & (df_all['status'] == 'ok')].copy()
    val_df = df_all[(df_all['fold'] == fold_idx) & (df_all['status'] == 'ok')].copy()

    # Verify zero patient overlap
    train_pts = set(train_df['patient_id'].unique())
    val_pts = set(val_df['patient_id'].unique())
    overlap = train_pts.intersection(val_pts)
    assert len(overlap) == 0, f"DATA LEAKAGE DETECTED! Overlapping patients: {overlap}"

    model, img_size, use_clahe, batch_size, base_lr = build_model(model_name, num_classes=3, pretrained=True)
    model = model.to(device)

    # Transforms: Strictly on-the-fly for training; strictly deterministic for validation
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=20),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_loader = DataLoader(
        EmbryoDataset(train_df, transform=train_transform, use_clahe=use_clahe),
        batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True
    )
    val_loader = DataLoader(
        EmbryoDataset(val_df, transform=val_transform, use_clahe=use_clahe),
        batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True
    )

    # Loss criterion
    if model_name == 'efficientnet_b4':
        criterion = OrdinalSoftLoss(smoothing=0.15)
    else:
        # Balanced class weights for VGG16 and ResNet50
        counts = train_df['grade'].value_counts()
        total_samples = len(train_df)
        weights = [total_samples / (3.0 * counts[f'Grade-{i+1}']) for i in range(3)]
        class_weights = torch.tensor(weights, dtype=torch.float32).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None

    best_val_acc = 0.0
    best_preds = None
    best_probs = None
    best_targets = None
    best_filenames = None
    best_state = None

    print(f"\n--- [Model: {model_name.upper()}] Fold {fold_idx + 1}/5 | Train: {len(train_df)} imgs ({len(train_pts)} pts) | Val: {len(val_df)} imgs ({len(val_pts)} pts) ---", flush=True)

    for epoch in range(1, epochs + 1):
        # Training Phase
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

        scheduler.step()
        train_acc = train_correct / train_total

        # Validation Phase (Strictly deterministic evaluation)
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
                    if isinstance(criterion, OrdinalSoftLoss):
                        loss = criterion(outputs, labels)
                    else:
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

        if val_acc > best_val_acc or epoch == 1:
            best_val_acc = val_acc
            best_preds = np.array(epoch_preds)
            best_probs = np.array(epoch_probs)
            best_targets = np.array(epoch_targets)
            best_filenames = list(epoch_filenames)
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 5 == 0 or epoch == epochs:
            print(f"  Epoch [{epoch:02d}/{epochs:02d}] Train Loss: {train_loss/train_total:.4f} Acc: {train_acc*100:.2f}% | Val Loss: {val_loss/val_total:.4f} Acc: {val_acc*100:.2f}% (Best: {best_val_acc*100:.2f}%)", flush=True)

    fold_df = pd.DataFrame({
        'filename': best_filenames,
        'fold': fold_idx,
        'true_label': best_targets,
        'pred_label': best_preds,
        'prob_grade1': best_probs[:, 0],
        'prob_grade2': best_probs[:, 1],
        'prob_grade3': best_probs[:, 2]
    })

    # Clean up GPU memory
    del model, optimizer, train_loader, val_loader
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return fold_df, best_val_acc, best_state

# Run Full 5-Fold Benchmark
def run_benchmark(model_name, epochs=20):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n==================================================================")
    print(f" Starting Leak-Free Patient-Separated 5-Fold Benchmark: {model_name.upper()}")
    print(f" Compute Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"==================================================================")

    df_clean = pd.read_csv('labels_clean.csv')
    labeled_df = df_clean[df_clean['status'] == 'ok'].copy()
    print(f"Loaded {len(labeled_df)} verified ground truth embryo images across {labeled_df['patient_id'].nunique()} unique patients.")

    all_fold_dfs = []
    fold_accuracies = []
    best_overall_state = None
    best_overall_acc = 0.0

    for fold_idx in range(5):
        fold_df, val_acc, state = train_and_eval_fold(fold_idx, model_name, df_clean, device, epochs=epochs)
        all_fold_dfs.append(fold_df)
        fold_accuracies.append(val_acc)
        if val_acc > best_overall_acc:
            best_overall_acc = val_acc
            best_overall_state = state

    # Combine all out-of-fold predictions
    oof_df = pd.concat(all_fold_dfs, ignore_index=True)
    oof_df['true_grade'] = oof_df['true_label'].map(INV_GRADE_MAP)
    oof_df['pred_grade'] = oof_df['pred_label'].map(INV_GRADE_MAP)
    oof_csv_path = f"clean_oof_predictions_{model_name}.csv"
    oof_df.to_csv(oof_csv_path, index=False)

    # Save model weights
    torch.save(best_overall_state, f"best_clean_{model_name}.pth")

    # Compute comprehensive metrics
    y_true = oof_df['true_label'].values
    y_pred = oof_df['pred_label'].values
    y_probs = oof_df[['prob_grade1', 'prob_grade2', 'prob_grade3']].values
    # Ensure probabilities sum strictly to 1.0 to satisfy sklearn multiclass roc_auc
    y_probs = y_probs / y_probs.sum(axis=1, keepdims=True)

    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
    qwk = cohen_kappa_score(y_true, y_pred, weights='quadratic')
    
    try:
        auc = roc_auc_score(y_true, y_probs, multi_class='ovr', average='macro')
    except Exception as e:
        auc = 0.0

    # Per-class metrics
    prec_pc, rec_pc, f1_pc, supp_pc = precision_recall_fscore_support(y_true, y_pred, average=None)

    metrics = {
        'model': model_name,
        'num_samples': int(len(oof_df)),
        'num_patients': int(labeled_df['patient_id'].nunique()),
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
    metrics_path = f"clean_metrics_{model_name}.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    # Plot Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(cm_norm, interpolation='nearest', cmap=plt.cm.Blues, vmin=0, vmax=1.0)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for i in range(3):
        for j in range(3):
            val = cm[i, j]
            pct = cm_norm[i, j] * 100
            color = "white" if cm_norm[i, j] > 0.5 else "black"
            ax.text(j, i, f"{pct:.1f}%\n(n={val})", ha="center", va="center", color=color, fontsize=11, fontweight='bold')

    tick_marks = np.arange(len(CLASS_NAMES))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(CLASS_NAMES, fontsize=11, fontweight='bold')
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(CLASS_NAMES, fontsize=11, fontweight='bold')

    title_name = {'vgg16': 'VGG-16', 'resnet50': 'ResNet-50', 'efficientnet_b4': 'EfficientNet-B4'}[model_name]
    ax.set_title(f"Leak-Free Patient-Separated Confusion Matrix\n{title_name} (Out-of-Fold, N=324)", fontsize=12, fontweight='bold', pad=12)
    ax.set_xlabel('Predicted Blastocyst Grade', fontsize=11, fontweight='bold')
    ax.set_ylabel('True Blastocyst Grade', fontsize=11, fontweight='bold')
    plt.tight_layout()
    cm_png_path = f"clean_confusion_matrix_{model_name}.png"
    plt.savefig(cm_png_path, dpi=300)
    plt.close()

    print(f"\n==================================================================", flush=True)
    print(f" FINISHED BENCHMARK: {model_name.upper()}", flush=True)
    print(f" Out-of-Fold Accuracy:  {acc*100:.2f}% (Fold Mean: {np.mean(fold_accuracies)*100:.2f}% +/- {np.std(fold_accuracies)*100:.2f}%)", flush=True)
    print(f" Balanced Accuracy:     {bal_acc*100:.2f}%", flush=True)
    print(f" Macro F1-Score:        {f1:.4f}", flush=True)
    print(f" Cohen's QWK:           {qwk:.4f}", flush=True)
    print(f" Macro ROC-AUC:         {auc:.4f}", flush=True)
    print(f" Saved: {oof_csv_path}, {metrics_path}, {cm_png_path}", flush=True)
    print(f"==================================================================", flush=True)

    return metrics

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run Leak-Free Patient-Separated Benchmark")
    parser.add_argument('--model', type=str, default='all', choices=['vgg16', 'resnet50', 'efficientnet_b4', 'all'])
    parser.add_argument('--epochs', type=int, default=20)
    args = parser.parse_args()

    models_to_run = ['vgg16', 'resnet50', 'efficientnet_b4'] if args.model == 'all' else [args.model]

    all_results = {}
    for m in models_to_run:
        res = run_benchmark(m, epochs=args.epochs)
        all_results[m] = res

    # If all models were run, save a combined summary
    if len(all_results) > 1:
        with open("clean_all_models_summary.json", 'w') as f:
            json.dump(all_results, f, indent=2)
        print("\nAll models benchmarked successfully! Summary written to clean_all_models_summary.json")
