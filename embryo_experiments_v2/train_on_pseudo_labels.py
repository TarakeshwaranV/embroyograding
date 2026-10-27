"""
Train Model on Graded Remaining Dataset and Evaluate on Clean Ground Truth Benchmark
=====================================================================================
Trains ResNet-50 on the newly graded remaining dataset (1,943 novel patient images)
and tests on the strictly held-out, verified ground-truth benchmark (324 images across 133 patients).

Zero patient leakage:
  - Training Set: Graded images belonging strictly to novel patients (no patient overlap with GT).
  - Test Set: The original 324 verified ground truth embryo images.
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
from torchvision import transforms

from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_recall_fscore_support,
    cohen_kappa_score, roc_auc_score, confusion_matrix
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)

from models import get_model
from dataset import get_transforms, CLAHETransform, GRADE_MAP, INV_GRADE_MAP, CLASS_NAMES

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

class EmbryoDataset(Dataset):
    def __init__(self, df, img_size=224, is_training=False, use_clahe=False):
        self.df = df.reset_index(drop=True)
        self.is_training = is_training
        self.use_clahe = use_clahe
        self.clahe = CLAHETransform() if use_clahe else None
        self.transform = get_transforms(img_size=img_size, is_training=is_training)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row['file_path']
        img = Image.open(img_path).convert('RGB')
        if self.use_clahe:
            img = self.clahe(img)
        tensor = self.transform(img)
        label = GRADE_MAP[row['grade']]
        return tensor, torch.tensor(label, dtype=torch.long)

def plot_confusion_matrix(cm, class_names, out_path, title='Confusion Matrix'):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names, yticklabels=class_names,
           title=title,
           ylabel='True Embryologist Grade',
           xlabel='Predicted Model Grade')

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

def main():
    seed_everything(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("=" * 75)
    print("   TRAINING ON GRADED DATASET & EVALUATING ON CLEAN GROUND TRUTH")
    print(f"   Execution Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print("=" * 75)

    # 1. Load Ground Truth Test Set (The 324 verified images from labels_clean.csv)
    labels_clean_path = os.path.join(PROJECT_ROOT, 'labels_clean.csv')
    df_clean = pd.read_csv(labels_clean_path)
    gt_test_df = df_clean[df_clean['status'] == 'ok'].dropna(subset=['grade']).copy()
    images_dir = os.path.join(PROJECT_ROOT, 'Images')
    gt_test_df['file_path'] = gt_test_df['filename'].apply(lambda fn: os.path.join(images_dir, fn))
    # verify images exist
    gt_test_df = gt_test_df[gt_test_df['file_path'].apply(os.path.exists)].reset_index(drop=True)
    print(f"\n[1] Verified Ground Truth Test Set: {len(gt_test_df)} images across {gt_test_df['patient_id'].nunique()} patients")
    print("    Class distribution in Test Set:")
    for c in CLASS_NAMES:
        print(f"      {c}: {(gt_test_df['grade'] == c).sum()} images")

    # 2. Load Graded Remaining Dataset
    graded_csv_path = os.path.join(BASE_DIR, 'predictions', 'graded_unlabeled_slices.csv')
    if not os.path.exists(graded_csv_path):
        raise FileNotFoundError(f"Graded slices CSV not found at {graded_csv_path}.")

    graded_df = pd.read_csv(graded_csv_path)

    # Strict Zero-Leakage: Exclude any patient present in the Ground Truth Test Set!
    test_patient_ids = set(gt_test_df['patient_id'])
    novel_graded_df = graded_df[~graded_df['patient_id'].isin(test_patient_ids)].copy()
    novel_graded_df['grade'] = novel_graded_df['final_predicted_grade']
    print(f"\n[2] Graded Training Candidates (Novel Patients Only): {len(novel_graded_df)} images across {novel_graded_df['patient_id'].nunique()} patients")
    print("    Graded Novel Patients Class Distribution:")
    for c in CLASS_NAMES:
        print(f"      {c}: {(novel_graded_df['grade'] == c).sum()} images")

    # Train / Val Split on novel training patients (85% train, 15% val for early stopping)
    pids = list(novel_graded_df['patient_id'].unique())
    random.shuffle(pids)
    split_idx = int(0.85 * len(pids))
    train_pids = set(pids[:split_idx])
    val_pids = set(pids[split_idx:])

    train_split = novel_graded_df[novel_graded_df['patient_id'].isin(train_pids)].reset_index(drop=True)
    val_split = novel_graded_df[novel_graded_df['patient_id'].isin(val_pids)].reset_index(drop=True)

    print(f"\n[3] Experimental Setup:")
    print(f"    Training Set:   {len(train_split)} images ({len(train_pids)} patients)")
    print(f"    Validation Set: {len(val_split)} images ({len(val_pids)} patients)")
    print(f"    Held-Out Test:  {len(gt_test_df)} images ({len(test_patient_ids)} patients - 100% UNSEEN GROUND TRUTH)")

    # DataLoaders (num_workers=0 for Windows compatibility)
    train_dataset = EmbryoDataset(train_split, img_size=224, is_training=True, use_clahe=True)
    val_dataset = EmbryoDataset(val_split, img_size=224, is_training=False, use_clahe=True)
    test_dataset = EmbryoDataset(gt_test_df, img_size=224, is_training=False, use_clahe=True)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0, pin_memory=True)

    # 4. Initialize Pretrained Model (ResNet-50)
    print(f"\n[4] Initializing ResNet-50 Architecture...")
    model = get_model('resnet50', num_classes=3, pretrained=True).to(device)

    # Safe class weighting
    counts = np.bincount([GRADE_MAP[g] for g in train_split['grade']], minlength=3)
    weights = torch.tensor([sum(counts) / (3.0 * max(c, 1)) for c in counts], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=8, eta_min=1e-6)

    # 5. Training Loop
    epochs = 8
    best_val_acc = 0.0
    best_weights = None
    save_ckpt_path = os.path.join(BASE_DIR, 'checkpoints', 'best_pseudo_labeled_resnet50.pth')

    print(f"\n[5] Fine-Tuning ResNet-50 on Graded Dataset ({epochs} epochs)...")
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
        with torch.no_grad():
            for imgs, targets in val_loader:
                imgs, targets = imgs.to(device), targets.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * len(targets)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == targets).sum().item()
                val_total += len(targets)

        val_loss /= val_total
        val_acc = (val_correct / val_total) * 100

        if val_acc > best_val_acc or best_weights is None:
            best_val_acc = val_acc
            best_weights = model.state_dict().copy()
            torch.save(best_weights, save_ckpt_path)

        print(f"    Epoch {epoch:2d}/{epochs:2d} | Train Loss: {train_loss:.4f} Acc: {train_acc:5.2f}% | Val Loss: {val_loss:.4f} Acc: {val_acc:5.2f}% {'[BEST]' if val_acc == best_val_acc else ''}")

    # 6. Strict Evaluation on Held-Out Ground Truth Test Set
    print("\n" + "=" * 75)
    print("   EVALUATING ON HELD-OUT VERIFIED GROUND TRUTH BENCHMARK (324 IMAGES)")
    print("=" * 75)

    model.load_state_dict(best_weights)
    model.eval()

    all_targets = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for imgs, targets in test_loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            probs = F.softmax(outputs, dim=1).cpu().numpy()
            preds = probs.argmax(axis=1)

            all_targets.extend(targets.numpy())
            all_preds.extend(preds)
            all_probs.extend(probs)

    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)

    acc = accuracy_score(y_true, y_pred) * 100
    bal_acc = balanced_accuracy_score(y_true, y_pred) * 100
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    qwk = cohen_kappa_score(y_true, y_pred, weights='quadratic')

    try:
        y_true_onehot = np.eye(3)[y_true]
        roc_auc = roc_auc_score(y_true_onehot, y_prob, average='macro', multi_class='ovr') * 100
    except Exception:
        roc_auc = 0.0

    cm = confusion_matrix(y_true, y_pred)

    print(f"\n   TEST RESULTS ON CLEAN GROUND TRUTH:")
    print(f"   -----------------------------------")
    print(f"   Overall Test Accuracy:        {acc:.2f}%")
    print(f"   Balanced Accuracy:            {bal_acc:.2f}%")
    print(f"   Macro-Precision:              {p_macro*100:.2f}%")
    print(f"   Macro-Recall:                 {r_macro*100:.2f}%")
    print(f"   Macro-F1 Score:               {f1_macro*100:.2f}%")
    print(f"   Quadratic Weighted Kappa:     {qwk:.4f}")
    print(f"   Macro Multiclass ROC-AUC:     {roc_auc:.2f}%")

    # Per-Class Metrics
    p_per, r_per, f1_per, _ = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
    print("\n   Per-Class Performance on Clean Ground Truth:")
    for idx, c in enumerate(CLASS_NAMES):
        print(f"     {c:10s} - Precision: {p_per[idx]*100:5.2f}% | Recall: {r_per[idx]*100:5.2f}% | F1: {f1_per[idx]*100:5.2f}%")

    # Save Confusion Matrix Plot
    plot_cm_path = os.path.join(BASE_DIR, 'plots', 'cm_pseudo_labeled_model_on_gt.png')
    plot_confusion_matrix(cm, CLASS_NAMES, plot_cm_path, title='Graded Dataset Model Evaluated on Ground Truth')
    print(f"\n   Saved Confusion Matrix plot: {plot_cm_path}")

    # Save Metrics JSON
    results = {
        'model': 'ResNet-50 trained on Graded Novel Patient Data',
        'training_samples': len(train_split),
        'test_samples': len(gt_test_df),
        'test_accuracy': float(acc),
        'balanced_accuracy': float(bal_acc),
        'macro_precision': float(p_macro * 100),
        'macro_recall': float(r_macro * 100),
        'macro_f1': float(f1_macro * 100),
        'quadratic_weighted_kappa': float(qwk),
        'macro_roc_auc': float(roc_auc),
        'confusion_matrix': cm.tolist(),
        'per_class': {
            CLASS_NAMES[i]: {
                'precision': float(p_per[i] * 100),
                'recall': float(r_per[i] * 100),
                'f1': float(f1_per[i] * 100)
            } for i in range(3)
        }
    }
    metrics_path = os.path.join(BASE_DIR, 'metrics', 'pseudo_labeled_model_gt_test_results.json')
    with open(metrics_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"   Saved Test Results JSON: {metrics_path}")

    # Save OOF predictions on GT test set
    gt_test_df['predicted_grade'] = [INV_GRADE_MAP[p] for p in y_pred]
    gt_test_df['conf_p1'] = y_prob[:, 0]
    gt_test_df['conf_p2'] = y_prob[:, 1]
    gt_test_df['conf_p3'] = y_prob[:, 2]
    gt_test_df.to_csv(os.path.join(BASE_DIR, 'predictions', 'pseudo_model_gt_test_predictions.csv'), index=False)
    print("=" * 75 + "\n")

if __name__ == '__main__':
    main()
