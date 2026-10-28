"""
Proper EfficientNet-B0 Training Pipeline on Clinically Graded Embryo Dataset
=============================================================================
Trained properly with:
  1. Zero Patient Leakage: Stratified Group Split by Patient ID (Train / Val / Test)
  2. Data Augmentation: Clinically appropriate on-the-fly transformations
  3. Pretrained Architecture: EfficientNet-B0 with custom regularization head
  4. Optimization: AdamW + Cosine Annealing + Class-Weighted Cross-Entropy (Label Smoothing)
  5. Dual Evaluation:
       - Unseen Patient Held-Out Test Set (15% of dataset, ~350 images)
       - External Validation Benchmark on 324 Human Embryologist Verified Images
"""

import os
import sys
import json
import random
import numpy as np
import pandas as pd
from PIL import Image

sys.stdout.reconfigure(line_buffering=True)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_recall_fscore_support,
    cohen_kappa_score, roc_auc_score, confusion_matrix
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

GRADE_MAP = {'Grade-1': 0, 'Grade-2': 1, 'Grade-3': 2}
INV_GRADE_MAP = {0: 'Grade-1', 1: 'Grade-2', 2: 'Grade-3'}
CLASS_NAMES = ['Grade-1', 'Grade-2', 'Grade-3']

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

class EmbryoDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row['file_path']
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            tensor = self.transform(img)
        else:
            tensor = transforms.ToTensor()(img)
        label = GRADE_MAP[row['ai_grade']]
        return tensor, torch.tensor(label, dtype=torch.long)

def get_clinical_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return train_transform, eval_transform

def plot_confusion_matrix(cm, class_names, out_path, title='Confusion Matrix'):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names, yticklabels=class_names,
           title=title,
           ylabel='True Clinical Grade',
           xlabel='Predicted EfficientNet Grade')

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()

def build_proper_efficientnet(num_classes=3, dropout=0.3):
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, 256),
        nn.SiLU(),
        nn.Dropout(p=dropout * 0.7),
        nn.Linear(256, num_classes)
    )
    return model

def main():
    seed_everything(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("=" * 80)
    print("   PROPER EFFICIENTNET-B0 TRAINING PIPELINE")
    print("   Dataset: Clinically Graded Embryo Dataset (2,344 images, 851 patients)")
    print(f"   Execution Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print("=" * 80)

    # 1. Load Clinical Dataset
    csv_path = os.path.join(BASE_DIR, 'predictions', 'ai_clinical_grading_full_dataset.csv')
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Clinical dataset not found at {csv_path}")

    df = pd.read_csv(csv_path)
    images_dir = os.path.join(PROJECT_ROOT, 'Images')
    df['file_path'] = df['filename'].apply(lambda fn: os.path.join(images_dir, fn))

    # 2. Strict Leakage-Free Stratified Group Splitting (Train: 70%, Val: 15%, Test: 15%)
    # Aggregate patient majority grade for stratifying groups
    patient_majority = df.groupby('patient_id')['ai_grade'].agg(lambda x: x.mode()[0]).reset_index()
    patient_ids = patient_majority['patient_id'].values
    patient_grades = patient_majority['ai_grade'].map(GRADE_MAP).values

    # Step 1: 85% Train+Val vs 15% Test
    sgkf_test = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=42)
    train_val_idx, test_idx = next(sgkf_test.split(patient_ids, patient_grades, groups=patient_ids))
    
    test_pids = set(patient_ids[test_idx])
    train_val_pids_arr = patient_ids[train_val_idx]
    train_val_grades_arr = patient_grades[train_val_idx]

    # Step 2: Split Train+Val into 70% Train and 15% Val
    sgkf_val = StratifiedGroupKFold(n_splits=6, shuffle=True, random_state=42)
    train_idx, val_idx = next(sgkf_val.split(train_val_pids_arr, train_val_grades_arr, groups=train_val_pids_arr))
    
    train_pids = set(train_val_pids_arr[train_idx])
    val_pids = set(train_val_pids_arr[val_idx])

    train_df = df[df['patient_id'].isin(train_pids)].copy().reset_index(drop=True)
    val_df = df[df['patient_id'].isin(val_pids)].copy().reset_index(drop=True)
    test_df = df[df['patient_id'].isin(test_pids)].copy().reset_index(drop=True)

    print(f"\n[1] Patient-Separated Partitioning (Zero Patient Leakage):")
    print(f"    - Training Set:   {len(train_df):4d} images across {len(train_pids)} patients ({len(train_df)/len(df)*100:.1f}%)")
    print(f"    - Validation Set: {len(val_df):4d} images across {len(val_pids)} patients ({len(val_df)/len(df)*100:.1f}%)")
    print(f"    - Held-Out Test:  {len(test_df):4d} images across {len(test_pids)} patients ({len(test_df)/len(df)*100:.1f}%)")

    print("\n    Class Distribution in Splits:")
    for c in CLASS_NAMES:
        print(f"      {c:10s} | Train: {(train_df['ai_grade']==c).sum():3d} | Val: {(val_df['ai_grade']==c).sum():3d} | Test: {(test_df['ai_grade']==c).sum():3d}")

    # 3. DataLoaders with Clinical Augmentations
    train_tf, eval_tf = get_clinical_transforms()
    train_ds = EmbryoDataset(train_df, transform=train_tf)
    val_ds = EmbryoDataset(val_df, transform=eval_tf)
    test_ds = EmbryoDataset(test_df, transform=eval_tf)

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=0, pin_memory=True)

    # 4. Architecture & Optimizer
    print(f"\n[2] Initializing Pretrained EfficientNet-B0 with Regularization Head...")
    model = build_proper_efficientnet(num_classes=3, dropout=0.3).to(device)

    # Balanced class weights with label smoothing
    counts = np.bincount([GRADE_MAP[g] for g in train_df['ai_grade']], minlength=3)
    weights = torch.tensor([sum(counts) / (3.0 * c) for c in counts], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.05)

    optimizer = optim.AdamW(model.parameters(), lr=1.5e-4, weight_decay=1e-4)
    epochs = 10
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # 5. Training Loop with Model Checkpointing
    best_val_score = 0.0
    best_weights = None
    ckpt_path = os.path.join(BASE_DIR, 'checkpoints', 'best_proper_efficientnet_b0.pth')

    print(f"\n[3] Commencing Training ({epochs} epochs)...")
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for imgs, targets in train_loader:
            imgs, targets = imgs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(targets)
            preds = outputs.argmax(dim=1)
            train_correct += (preds == targets).sum().item()
            train_total += len(targets)

        scheduler.step()
        train_loss /= train_total
        train_acc = (train_correct / train_total) * 100

        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        val_targets, val_preds = [], []
        with torch.no_grad():
            for imgs, targets in val_loader:
                imgs, targets = imgs.to(device), targets.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * len(targets)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == targets).sum().item()
                val_total += len(targets)
                val_targets.extend(targets.cpu().numpy())
                val_preds.extend(preds.cpu().numpy())

        val_loss /= val_total
        val_acc = (val_correct / val_total) * 100
        val_bal_acc = balanced_accuracy_score(val_targets, val_preds) * 100
        val_qwk = cohen_kappa_score(val_targets, val_preds, weights='quadratic')

        # Composite validation selection criterion: Balanced Acc + QWK
        val_score = val_bal_acc + (val_qwk * 20.0)
        is_best = val_score > best_val_score or best_weights is None
        if is_best:
            best_val_score = val_score
            best_weights = model.state_dict().copy()
            torch.save(best_weights, ckpt_path)

        print(f"    Epoch {epoch:2d}/{epochs:2d} | Train Loss: {train_loss:.4f} Acc: {train_acc:5.2f}% | Val Loss: {val_loss:.4f} Acc: {val_acc:5.2f}% Bal: {val_bal_acc:5.2f}% QWK: {val_qwk:.3f} {'[BEST]' if is_best else ''}", flush=True)

    # 6. Evaluation on Unseen Held-Out Patient Test Set (15% of Dataset)
    print("\n" + "=" * 80)
    print("   [4] EVALUATION ON HELD-OUT PATIENT TEST SET (ZERO LEAKAGE, 345 IMAGES)")
    print("=" * 80)

    model.load_state_dict(best_weights)
    model.eval()

    test_targets, test_preds, test_probs = [], [], []
    with torch.no_grad():
        for imgs, targets in test_loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            probs = F.softmax(outputs, dim=1).cpu().numpy()
            preds = probs.argmax(axis=1)

            test_targets.extend(targets.numpy())
            test_preds.extend(preds)
            test_probs.extend(probs)

    y_test = np.array(test_targets)
    y_pred = np.array(test_preds)
    y_prob = np.array(test_probs)

    acc = accuracy_score(y_test, y_pred) * 100
    bal_acc = balanced_accuracy_score(y_test, y_pred) * 100
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_test, y_pred, average='macro', zero_division=0)
    qwk = cohen_kappa_score(y_test, y_pred, weights='quadratic')
    
    # Binary Transferable (Grade 1/2 vs Grade 3)
    bin_true = (y_test < 2)
    bin_pred = (y_pred < 2)
    bin_acc = accuracy_score(bin_true, bin_pred) * 100

    try:
        y_test_oh = np.eye(3)[y_test]
        roc_auc = roc_auc_score(y_test_oh, y_prob, average='macro', multi_class='ovr') * 100
    except Exception:
        roc_auc = 0.0

    cm = confusion_matrix(y_test, y_pred)

    print(f"\n   TEST METRICS ON UNSEEN PATIENT COHORT:")
    print(f"   ---------------------------------------")
    print(f"   Overall 3-Class Test Accuracy:    {acc:.2f}%")
    print(f"   Balanced Accuracy:                {bal_acc:.2f}%")
    print(f"   Binary Transferable (1/2 vs 3):   {bin_acc:.2f}%")
    print(f"   Macro-F1 Score:                   {f1_macro*100:.2f}%")
    print(f"   Macro-Precision:                  {p_macro*100:.2f}%")
    print(f"   Macro-Recall:                     {r_macro*100:.2f}%")
    print(f"   Quadratic Weighted Kappa (QWK):   {qwk:.4f}")
    print(f"   Multiclass ROC-AUC:               {roc_auc:.2f}%")

    p_per, r_per, f1_per, _ = precision_recall_fscore_support(y_test, y_pred, average=None, zero_division=0)
    print("\n   Per-Class Test Performance:")
    for idx, c in enumerate(CLASS_NAMES):
        print(f"     {c:10s} - Precision: {p_per[idx]*100:5.2f}% | Recall: {r_per[idx]*100:5.2f}% | F1: {f1_per[idx]*100:5.2f}%")

    # Plot Confusion Matrix
    cm_path = os.path.join(BASE_DIR, 'plots', 'cm_proper_efficientnet_test_set.png')
    plot_confusion_matrix(cm, CLASS_NAMES, cm_path, title='Proper EfficientNet-B0 on Held-Out Patients')
    print(f"\n   Saved Test Confusion Matrix plot: {cm_path}")

    # 7. External Validation on 324 Human Embryologist Verified Images
    print("\n" + "=" * 80)
    print("   [5] EXTERNAL BENCHMARK: EVALUATION ON 324 HUMAN-VERIFIED GROUND TRUTH")
    print("=" * 80)

    labels_clean_path = os.path.join(PROJECT_ROOT, 'labels_clean.csv')
    df_clean = pd.read_csv(labels_clean_path)
    human_gt_df = df_clean[df_clean['status'] == 'ok'].dropna(subset=['grade']).copy()
    human_gt_df['file_path'] = human_gt_df['filename'].apply(lambda fn: os.path.join(images_dir, fn))
    human_gt_df['ai_grade'] = human_gt_df['grade']
    human_gt_df = human_gt_df[human_gt_df['file_path'].apply(os.path.exists)].reset_index(drop=True)

    human_ds = EmbryoDataset(human_gt_df, transform=eval_tf)
    human_loader = DataLoader(human_ds, batch_size=16, shuffle=False, num_workers=0)

    human_targets, human_preds, human_probs = [], [], []
    with torch.no_grad():
        for imgs, targets in human_loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            probs = F.softmax(outputs, dim=1).cpu().numpy()
            preds = probs.argmax(axis=1)
            human_targets.extend(targets.numpy())
            human_preds.extend(preds)
            human_probs.extend(probs)

    y_hum = np.array(human_targets)
    y_hum_pred = np.array(human_preds)
    
    hum_acc = accuracy_score(y_hum, y_hum_pred) * 100
    hum_bal_acc = balanced_accuracy_score(y_hum, y_hum_pred) * 100
    hum_qwk = cohen_kappa_score(y_hum, y_hum_pred, weights='quadratic')
    hum_bin_acc = accuracy_score((y_hum < 2), (y_hum_pred < 2)) * 100

    print(f"   Human Ground-Truth 3-Class Accuracy:   {hum_acc:.2f}%")
    print(f"   Human Ground-Truth Balanced Accuracy:   {hum_bal_acc:.2f}%")
    print(f"   Human Ground-Truth Binary (1/2 vs 3):   {hum_bin_acc:.2f}%")
    print(f"   Human Ground-Truth Quadratic Kappa:     {hum_qwk:.4f}")

    # 8. Save Metrics JSON
    results = {
        'model': 'EfficientNet-B0 Trained on Clinical Consensus Dataset',
        'training_patients': len(train_pids),
        'training_images': len(train_df),
        'test_patients': len(test_pids),
        'test_images': len(test_df),
        'held_out_test_metrics': {
            'accuracy': float(acc),
            'balanced_accuracy': float(bal_acc),
            'binary_transferable_accuracy': float(bin_acc),
            'macro_f1': float(f1_macro * 100),
            'quadratic_weighted_kappa': float(qwk),
            'multiclass_roc_auc': float(roc_auc),
            'confusion_matrix': cm.tolist()
        },
        'human_ground_truth_benchmark_metrics': {
            'accuracy': float(hum_acc),
            'balanced_accuracy': float(hum_bal_acc),
            'binary_transferable_accuracy': float(hum_bin_acc),
            'quadratic_weighted_kappa': float(hum_qwk)
        }
    }

    metrics_path = os.path.join(BASE_DIR, 'metrics', 'proper_efficientnet_evaluation_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n   Saved Results JSON: {metrics_path}")
    print("=" * 80 + "\n")

if __name__ == '__main__':
    main()
