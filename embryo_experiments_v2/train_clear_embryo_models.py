"""
Super-Ensemble Training on Clear Patient Embryo Dataset
=======================================================
Trains the Heterogeneous Super-Ensemble directly on the 851 High-Clarity Patient Embryos:
  - 1 Sharp, In-Focus, Auto-Centered, CLAHE-Enhanced Image Per Patient
  - Strict Zero Patient Data Leakage (Stratified Split on Patients)
  - 4 Deep Architectures: ConvNeXt-Tiny, DenseNet-121, ResNet-50, EfficientNet-B0
  - 5-View Test-Time Augmentation (TTA) + Weighted Soft Probability Fusion
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
from torchvision.models import (
    convnext_tiny, ConvNeXt_Tiny_Weights,
    densenet121, DenseNet121_Weights,
    resnet50, ResNet50_Weights,
    efficientnet_b0, EfficientNet_B0_Weights
)

from sklearn.model_selection import StratifiedKFold
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

class ClearEmbryoDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row['processed_file_path']).convert('RGB')
        if self.transform:
            tensor = self.transform(img)
        else:
            tensor = transforms.ToTensor()(img)
        label = GRADE_MAP[row['ai_grade']]
        return tensor, torch.tensor(label, dtype=torch.long)

def get_transforms(img_size=256):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=20),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        normalize
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        normalize
    ])
    tta_transforms = [
        transforms.Compose([transforms.Resize((img_size, img_size)), transforms.ToTensor(), normalize]),
        transforms.Compose([transforms.Resize((img_size, img_size)), transforms.RandomHorizontalFlip(p=1.0), transforms.ToTensor(), normalize]),
        transforms.Compose([transforms.Resize((img_size, img_size)), transforms.RandomVerticalFlip(p=1.0), transforms.ToTensor(), normalize]),
        transforms.Compose([transforms.Resize((img_size, img_size)), transforms.RandomRotation(degrees=(15, 15)), transforms.ToTensor(), normalize]),
        transforms.Compose([transforms.Resize((img_size, img_size)), transforms.RandomRotation(degrees=(-15, -15)), transforms.ToTensor(), normalize]),
    ]
    return train_transform, eval_transform, tta_transforms

def build_convnext():
    model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT)
    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, 256),
        nn.GELU(),
        nn.Dropout(p=0.2),
        nn.Linear(256, 3)
    )
    return model

def build_densenet():
    model = densenet121(weights=DenseNet121_Weights.DEFAULT)
    in_features = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, 256),
        nn.SiLU(),
        nn.Dropout(p=0.2),
        nn.Linear(256, 3)
    )
    return model

def build_resnet():
    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, 256),
        nn.SiLU(),
        nn.Dropout(p=0.2),
        nn.Linear(256, 3)
    )
    return model

def build_efficientnet():
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, 256),
        nn.SiLU(),
        nn.Dropout(p=0.2),
        nn.Linear(256, 3)
    )
    return model

def train_model(name, model, train_loader, val_loader, class_weights, device, epochs=12, lr=1.8e-4):
    print(f"\n---> Training Architecture: {name} ({epochs} epochs, lr={lr})...")
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())

    best_score = -1.0
    best_weights = None
    best_train_acc = 0.0
    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for imgs, targets in train_loader:
            imgs, targets = imgs.to(device), targets.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                outputs = model(imgs)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * len(targets)
            train_correct += (outputs.argmax(dim=1) == targets).sum().item()
            train_total += len(targets)

        scheduler.step()
        train_loss /= train_total
        train_acc = (train_correct / train_total) * 100

        # Validation
        model.eval()
        val_loss, val_targets, val_preds = 0.0, [], []
        with torch.no_grad():
            for imgs, targets in val_loader:
                imgs, targets = imgs.to(device), targets.to(device)
                with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                    outputs = model(imgs)
                    loss = criterion(outputs, targets)
                val_loss += loss.item() * len(targets)
                val_targets.extend(targets.cpu().numpy())
                val_preds.extend(outputs.argmax(dim=1).cpu().numpy())

        val_loss /= len(val_targets)
        val_acc = accuracy_score(val_targets, val_preds) * 100
        val_bal = balanced_accuracy_score(val_targets, val_preds) * 100
        val_qwk = cohen_kappa_score(val_targets, val_preds, weights='quadratic')
        score = val_bal + (val_qwk * 20.0)

        is_best = score > best_score
        if is_best:
            best_score = score
            best_train_acc = train_acc
            best_val_acc = val_acc
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        print(f"     Epoch {epoch:2d}/{epochs:2d} | Train: Loss {train_loss:.4f} Acc {train_acc:5.2f}% | Val: Loss {val_loss:.4f} Acc {val_acc:5.2f}% Bal {val_bal:5.2f}% QWK {val_qwk:.3f} {'[BEST]' if is_best else ''}", flush=True)

    model.load_state_dict(best_weights)
    model.to(device).eval()
    return model, best_train_acc, best_val_acc

def predict_tta(model, df, tta_transforms, device, batch_size=16):
    model.eval()
    all_view_probs = []
    for tf in tta_transforms:
        ds = ClearEmbryoDataset(df, transform=tf)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
        view_probs = []
        with torch.no_grad():
            for imgs, _ in loader:
                imgs = imgs.to(device)
                with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                    logits = model(imgs)
                probs = F.softmax(logits, dim=1).cpu().numpy()
                view_probs.append(probs)
        all_view_probs.append(np.concatenate(view_probs, axis=0))
    avg_probs = np.mean(all_view_probs, axis=0)
    return avg_probs

def plot_confusion_matrix(cm, class_names, out_path, title='Confusion Matrix'):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names, yticklabels=class_names,
           title=title,
           ylabel='True Clinical Grade',
           xlabel='Predicted Grade')

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

    print("=" * 80)
    print("   CLEAR PATIENT EMBRYO TRAINING PIPELINE (ZERO LEAKAGE)")
    print(f"   Execution Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print("=" * 80)

    manifest_path = os.path.join(BASE_DIR, 'predictions', 'clear_patient_embryos_manifest.csv')
    df = pd.read_csv(manifest_path)
    print(f"[1] Loaded {len(df)} Clear Patient Embryo images (1 image per patient).")

    # Patient-stratified split: 70% Train (595), 15% Val (128), 15% Test (128)
    skf_test = StratifiedKFold(n_splits=7, shuffle=True, random_state=42)
    train_val_idx, test_idx = next(skf_test.split(df, df['ai_grade']))

    train_val_df = df.iloc[train_val_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    skf_val = StratifiedKFold(n_splits=6, shuffle=True, random_state=42)
    train_idx, val_idx = next(skf_val.split(train_val_df, train_val_df['ai_grade']))

    train_df = train_val_df.iloc[train_idx].reset_index(drop=True)
    val_df = train_val_df.iloc[val_idx].reset_index(drop=True)

    print(f"\n[2] Patient Cohort Splits (Strict 0% Leakage):")
    print(f"    - Training:   {len(train_df):3d} patients ({len(train_df)/len(df)*100:.1f}%)")
    print(f"    - Validation: {len(val_df):3d} patients ({len(val_df)/len(df)*100:.1f}%)")
    print(f"    - Held-Out:   {len(test_df):3d} patients ({len(test_df)/len(df)*100:.1f}%)")

    train_tf, eval_tf, tta_tfs = get_transforms(img_size=256)
    train_ds = ClearEmbryoDataset(train_df, transform=train_tf)
    val_ds = ClearEmbryoDataset(val_df, transform=eval_tf)

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=0, pin_memory=True)

    counts = np.bincount([GRADE_MAP[g] for g in train_df['ai_grade']], minlength=3)
    class_weights = torch.tensor([sum(counts) / (3.0 * c) for c in counts], dtype=torch.float32).to(device)

    # 3. Train the 4 Architectures
    ckpt_dir = os.path.join(BASE_DIR, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)

    # ConvNeXt-Tiny
    m_conv = build_convnext().to(device)
    m_conv, train_acc_conv, val_acc_conv = train_model('ConvNeXt-Tiny', m_conv, train_loader, val_loader, class_weights, device, epochs=12, lr=1.8e-4)
    torch.save(m_conv.state_dict(), os.path.join(ckpt_dir, 'clear_convnext_tiny.pth'))

    # DenseNet-121
    m_dense = build_densenet().to(device)
    m_dense, train_acc_dense, val_acc_dense = train_model('DenseNet-121', m_dense, train_loader, val_loader, class_weights, device, epochs=12, lr=2.0e-4)
    torch.save(m_dense.state_dict(), os.path.join(ckpt_dir, 'clear_densenet121.pth'))

    # ResNet-50
    m_res = build_resnet().to(device)
    m_res, train_acc_res, val_acc_res = train_model('ResNet-50', m_res, train_loader, val_loader, class_weights, device, epochs=12, lr=1.5e-4)
    torch.save(m_res.state_dict(), os.path.join(ckpt_dir, 'clear_resnet50.pth'))

    # EfficientNet-B0
    m_eff = build_efficientnet().to(device)
    m_eff, train_acc_eff, val_acc_eff = train_model('EfficientNet-B0', m_eff, train_loader, val_loader, class_weights, device, epochs=12, lr=1.5e-4)
    torch.save(m_eff.state_dict(), os.path.join(ckpt_dir, 'clear_efficientnet_b0.pth'))

    # 4. TTA Probabilities on Validation & Ensemble Optimization
    print("\n" + "=" * 80)
    print("   [4] EXTRACTING 5-VIEW TTA PROBABILITIES ON VALIDATION SET...")
    print("=" * 80)
    p_val_conv = predict_tta(m_conv, val_df, tta_tfs, device)
    p_val_dense = predict_tta(m_dense, val_df, tta_tfs, device)
    p_val_res = predict_tta(m_res, val_df, tta_tfs, device)
    p_val_eff = predict_tta(m_eff, val_df, tta_tfs, device)

    y_val = np.array([GRADE_MAP[g] for g in val_df['ai_grade']])

    best_w = [0.25, 0.25, 0.25, 0.25]
    best_ens_acc = 0.0
    for w1 in np.linspace(0.1, 0.5, 5):
        for w2 in np.linspace(0.1, 0.5, 5):
            for w3 in np.linspace(0.1, 0.5, 5):
                w4 = max(0.0, 1.0 - (w1 + w2 + w3))
                if w4 < 0.05 or w4 > 0.6:
                    continue
                w_vec = np.array([w1, w2, w3, w4])
                w_vec /= w_vec.sum()
                ens_p = w_vec[0]*p_val_conv + w_vec[1]*p_val_dense + w_vec[2]*p_val_res + w_vec[3]*p_val_eff
                score = accuracy_score(y_val, ens_p.argmax(axis=1)) * 100
                if score > best_ens_acc:
                    best_ens_acc = score
                    best_w = w_vec

    print(f"    Optimized Ensemble Weights: ConvNeXt={best_w[0]:.2f}, DenseNet={best_w[1]:.2f}, ResNet={best_w[2]:.2f}, EffNet={best_w[3]:.2f}")
    print(f"    Clear Embryo Validation Ensemble Accuracy: {best_ens_acc:.2f}%")

    # 5. Evaluation on Held-Out Test Patients (128 images)
    print("\n" + "=" * 80)
    print("   [5] STRICT EVALUATION ON HELD-OUT PATIENT TEST SET (128 PATIENTS - ZERO LEAKAGE)")
    print("=" * 80)
    p_test_conv = predict_tta(m_conv, test_df, tta_tfs, device)
    p_test_dense = predict_tta(m_dense, test_df, tta_tfs, device)
    p_test_res = predict_tta(m_res, test_df, tta_tfs, device)
    p_test_eff = predict_tta(m_eff, test_df, tta_tfs, device)

    y_test = np.array([GRADE_MAP[g] for g in test_df['ai_grade']])

    # Individual Model Test Accuracies
    acc_conv = accuracy_score(y_test, p_test_conv.argmax(axis=1)) * 100
    acc_dense = accuracy_score(y_test, p_test_dense.argmax(axis=1)) * 100
    acc_res = accuracy_score(y_test, p_test_res.argmax(axis=1)) * 100
    acc_eff = accuracy_score(y_test, p_test_eff.argmax(axis=1)) * 100

    print(f"    - ConvNeXt-Tiny Test Acc:   {acc_conv:.2f}%")
    print(f"    - DenseNet-121  Test Acc:   {acc_dense:.2f}%")
    print(f"    - ResNet-50     Test Acc:   {acc_res:.2f}%")
    print(f"    - EfficientNet  Test Acc:   {acc_eff:.2f}%")

    # Super-Ensemble Prediction
    p_test_ens = best_w[0]*p_test_conv + best_w[1]*p_test_dense + best_w[2]*p_test_res + best_w[3]*p_test_eff
    preds_test = p_test_ens.argmax(axis=1)

    acc = accuracy_score(y_test, preds_test) * 100
    bal_acc = balanced_accuracy_score(y_test, preds_test) * 100
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_test, preds_test, average='macro', zero_division=0)
    qwk = cohen_kappa_score(y_test, preds_test, weights='quadratic')

    # Binary Transferable Viability (Grade 1/2 vs 3)
    bin_true = (y_test < 2)
    bin_pred = (preds_test < 2)
    bin_acc = accuracy_score(bin_true, bin_pred) * 100
    bin_auc = roc_auc_score(bin_true, p_test_ens[:, 0] + p_test_ens[:, 1]) * 100

    print(f"\n   CLEAR PATIENT TEST RESULTS (128 PATIENTS):")
    print(f"   ------------------------------------------")
    print(f"   3-Class Accuracy:                 {acc:.2f}%")
    print(f"   Balanced Accuracy:                {bal_acc:.2f}%")
    print(f"   Macro-F1 Score:                   {f1_macro*100:.2f}%")
    print(f"   Quadratic Weighted Kappa (QWK):   {qwk:.4f}")
    print(f"   Binary Viable (Grade 1/2 vs 3):   {bin_acc:.2f}%")
    print(f"   Binary Viable AUC:                {bin_auc:.2f}%")

    p_per, r_per, f1_per, _ = precision_recall_fscore_support(y_test, preds_test, average=None, zero_division=0)
    for idx, c in enumerate(CLASS_NAMES):
        print(f"     {c:10s} - Precision: {p_per[idx]*100:5.2f}% | Recall: {r_per[idx]*100:5.2f}% | F1: {f1_per[idx]*100:5.2f}%")

    # 6. Save Outputs
    plots_dir = os.path.join(BASE_DIR, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    cm = confusion_matrix(y_test, preds_test)
    cm_path = os.path.join(plots_dir, 'cm_clear_embryo_test.png')
    plot_confusion_matrix(cm, CLASS_NAMES, cm_path, title='Clear Patient Embryo Super-Ensemble Test')

    test_out_df = test_df.copy()
    test_out_df['pred_prob_g1'] = p_test_ens[:, 0]
    test_out_df['pred_prob_g2'] = p_test_ens[:, 1]
    test_out_df['pred_prob_g3'] = p_test_ens[:, 2]
    test_out_df['ensemble_predicted_grade'] = [INV_GRADE_MAP[p] for p in preds_test]
    test_out_df['is_correct'] = (test_out_df['ai_grade'] == test_out_df['ensemble_predicted_grade'])

    preds_dir = os.path.join(BASE_DIR, 'predictions')
    out_csv = os.path.join(preds_dir, 'test_clear_embryo_predictions.csv')
    test_out_df.to_csv(out_csv, index=False)

    avg_train_acc = (train_acc_conv + train_acc_dense + train_acc_res + train_acc_eff) / 4.0

    metrics_dir = os.path.join(BASE_DIR, 'metrics')
    out_json = os.path.join(metrics_dir, 'clear_embryo_benchmark_results.json')
    results = {
        "dataset": "851 High-Clarity Clear Patient Embryos",
        "protocol": "Stratified Zero Patient Data Leakage",
        "super_ensemble_metrics": {
            "training_accuracy_avg": float(avg_train_acc),
            "training_accuracy_peak": float(max(train_acc_conv, train_acc_dense, train_acc_res, train_acc_eff)),
            "validation_accuracy": float(best_ens_acc),
            "test_accuracy_3_class": float(acc),
            "balanced_accuracy": float(bal_acc),
            "macro_f1": float(f1_macro * 100),
            "quadratic_weighted_kappa": float(qwk),
            "binary_viability_accuracy": float(bin_acc),
            "binary_viability_auc": float(bin_auc),
            "confusion_matrix": cm.tolist()
        },
        "component_models": {
            "convnext_tiny": {"train_acc": float(train_acc_conv), "val_acc": float(val_acc_conv), "test_acc": float(acc_conv)},
            "densenet121":   {"train_acc": float(train_acc_dense), "val_acc": float(val_acc_dense), "test_acc": float(acc_dense)},
            "resnet50":      {"train_acc": float(train_acc_res), "val_acc": float(val_acc_res), "test_acc": float(acc_res)},
            "efficientnet_b0": {"train_acc": float(train_acc_eff), "val_acc": float(val_acc_eff), "test_acc": float(acc_eff)}
        }
    }
    with open(out_json, 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 80)
    print("   Clear Embryo Benchmark Complete:")
    print(f"   - Metrics JSON:     {out_json}")
    print(f"   - Predictions CSV:  {out_csv}")
    print(f"   - Confusion Matrix: {cm_path}")
    print("=" * 80 + "\n")

if __name__ == '__main__':
    main()
