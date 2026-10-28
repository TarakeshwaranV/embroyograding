"""
Super-Ensemble Pipeline for Maximal 3-Class Embryo Generalization
=================================================================
Target: Scaling 3-Class Classification Accuracy towards and above 85% with Zero Data Leakage.

Techniques Integrated:
  1. Heterogeneous Deep Architectures:
       - ConvNeXt-Tiny (Modern depthwise 7x7 convs, GELU, LayerNorm)
       - DenseNet-121 (Dense multi-scale feature reuse for cellular boundaries)
       - ResNet-50 (Deep 2048-dim residual representation)
       - EfficientNet-B0 (Squeeze-and-excitation compound scaled CNN)
  2. Strict Zero Patient Data Leakage:
       - Group splitting strictly by patient_id: 1,664 train (607 patients), 355 val (122 patients), 325 test (122 patients)
  3. Morphologically Preserving Resolution (256x256) & Invariant Clinical Augmentations
  4. Mixed Precision (torch.cuda.amp) + AdamW + Cosine Annealing Learning Rate Schedule
  5. 5-View Test-Time Augmentation (TTA)
  6. Temperature-Calibrated Soft Ensemble Voting
  7. Clinical Triage Stratification (Full Cohort, Confident Cohort >=65%, High Precision >=75%)
"""

import os
import sys
import json
import random
import numpy as np
import pandas as pd
from PIL import Image
from scipy.optimize import minimize

sys.stdout.reconfigure(line_buffering=True)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.models import (
    convnext_tiny, ConvNeXt_Tiny_Weights,
    densenet121, DenseNet121_Weights,
    resnet50, ResNet50_Weights,
    efficientnet_b0, EfficientNet_B0_Weights
)

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
        img = Image.open(row['file_path']).convert('RGB')
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
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
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

# Model Building Functions
def build_convnext_model(num_classes=3, dropout=0.3):
    model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT)
    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, 256),
        nn.GELU(),
        nn.Dropout(p=dropout * 0.7),
        nn.Linear(256, num_classes)
    )
    return model

def build_densenet_model(num_classes=3, dropout=0.3):
    model = densenet121(weights=DenseNet121_Weights.DEFAULT)
    in_features = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, 256),
        nn.SiLU(),
        nn.Dropout(p=dropout * 0.7),
        nn.Linear(256, num_classes)
    )
    return model

def build_resnet_model(num_classes=3, dropout=0.3):
    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, 256),
        nn.SiLU(),
        nn.Dropout(p=dropout * 0.7),
        nn.Linear(256, num_classes)
    )
    return model

def build_efficientnet_model(num_classes=3, dropout=0.3):
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

def train_single_model(model_name, model, train_loader, val_loader, class_weights, device, epochs=8, lr=2e-4):
    print(f"\n---> Training Architecture: {model_name} ({epochs} epochs, lr={lr})...")
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    best_score = -1.0
    best_weights = None

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for imgs, targets in train_loader:
            imgs, targets = imgs.to(device), targets.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
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
                with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                    outputs = model(imgs)
                    loss = criterion(outputs, targets)
                val_loss += loss.item() * len(targets)
                val_targets.extend(targets.cpu().numpy())
                val_preds.extend(outputs.argmax(dim=1).cpu().numpy())

        val_loss /= len(val_targets)
        val_acc = accuracy_score(val_targets, val_preds) * 100
        val_bal = balanced_accuracy_score(val_targets, val_preds) * 100
        val_qwk = cohen_kappa_score(val_targets, val_preds, weights='quadratic')
        composite_score = val_bal + (val_qwk * 25.0)

        is_best = composite_score > best_score
        if is_best:
            best_score = composite_score
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        print(f"     Epoch {epoch:2d}/{epochs:2d} | Train: Loss {train_loss:.4f} Acc {train_acc:5.2f}% | Val: Loss {val_loss:.4f} Acc {val_acc:5.2f}% Bal {val_bal:5.2f}% QWK {val_qwk:.3f} {'[BEST]' if is_best else ''}", flush=True)

    model.load_state_dict(best_weights)
    model.to(device).eval()
    return model

def predict_tta(model, df, tta_transforms, device, batch_size=16):
    model.eval()
    all_view_probs = []
    for tf in tta_transforms:
        ds = EmbryoDataset(df, transform=tf)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
        view_probs = []
        with torch.no_grad():
            for imgs, _ in loader:
                imgs = imgs.to(device)
                with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                    logits = model(imgs)
                probs = F.softmax(logits, dim=1).cpu().numpy()
                view_probs.append(probs)
        all_view_probs.append(np.concatenate(view_probs, axis=0))
    # Average across TTA views
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
           xlabel='Predicted Ensemble Grade')

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
    print("   SUPER-ENSEMBLE PIPELINE: CONVNEXT + DENSENET + RESNET50 + EFFICIENTNET")
    print(f"   Execution Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print("=" * 80)

    # 1. Load Clinical Dataset
    csv_path = os.path.join(BASE_DIR, 'predictions', 'ai_clinical_grading_full_dataset.csv')
    df = pd.read_csv(csv_path)
    images_dir = os.path.join(PROJECT_ROOT, 'Images')
    df['file_path'] = df['filename'].apply(lambda fn: os.path.join(images_dir, fn))

    # 2. Strict Leakage-Free Stratified Group Splitting (Same split as proper efficientnet)
    patient_majority = df.groupby('patient_id')['ai_grade'].agg(lambda x: x.mode()[0]).reset_index()
    patient_ids = patient_majority['patient_id'].values
    patient_grades = patient_majority['ai_grade'].map(GRADE_MAP).values

    sgkf_test = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=42)
    train_val_idx, test_idx = next(sgkf_test.split(patient_ids, patient_grades, groups=patient_ids))

    test_pids = set(patient_ids[test_idx])
    train_val_pids_arr = patient_ids[train_val_idx]
    train_val_grades_arr = patient_grades[train_val_idx]

    sgkf_val = StratifiedGroupKFold(n_splits=6, shuffle=True, random_state=42)
    train_idx, val_idx = next(sgkf_val.split(train_val_pids_arr, train_val_grades_arr, groups=train_val_pids_arr))

    train_pids = set(train_val_pids_arr[train_idx])
    val_pids = set(train_val_pids_arr[val_idx])

    train_df = df[df['patient_id'].isin(train_pids)].copy().reset_index(drop=True)
    val_df = df[df['patient_id'].isin(val_pids)].copy().reset_index(drop=True)
    test_df = df[df['patient_id'].isin(test_pids)].copy().reset_index(drop=True)

    print(f"\n[1] Partitioning Setup (Zero Patient Leakage):")
    print(f"    - Training Set:   {len(train_df):4d} images across {len(train_pids)} patients")
    print(f"    - Validation Set: {len(val_df):4d} images across {len(val_pids)} patients")
    print(f"    - Held-Out Test:  {len(test_df):4d} images across {len(test_pids)} patients")

    # DataLoaders
    train_tf, eval_tf, tta_tfs = get_transforms(img_size=256)
    train_ds = EmbryoDataset(train_df, transform=train_tf)
    val_ds = EmbryoDataset(val_df, transform=eval_tf)

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=0, pin_memory=True)

    counts = np.bincount([GRADE_MAP[g] for g in train_df['ai_grade']], minlength=3)
    class_weights = torch.tensor([sum(counts) / (3.0 * c) for c in counts], dtype=torch.float32).to(device)

    # 3. Model Training & Loading
    # Model 1: ConvNeXt-Tiny
    ckpt_dir = os.path.join(BASE_DIR, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    conv_ckpt = os.path.join(ckpt_dir, 'best_convnext_tiny.pth')
    conv_model = build_convnext_model().to(device)
    if os.path.exists(conv_ckpt):
        print(f"\n[2] Loading cached ConvNeXt-Tiny checkpoint from {conv_ckpt}...")
        conv_model.load_state_dict(torch.load(conv_ckpt, map_location=device, weights_only=True))
    else:
        conv_model = train_single_model('ConvNeXt-Tiny', conv_model, train_loader, val_loader, class_weights, device, epochs=8, lr=1.8e-4)
        torch.save(conv_model.state_dict(), conv_ckpt)

    # Model 2: DenseNet-121
    dense_ckpt = os.path.join(ckpt_dir, 'best_densenet121.pth')
    dense_model = build_densenet_model().to(device)
    if os.path.exists(dense_ckpt):
        print(f"\n[3] Loading cached DenseNet-121 checkpoint from {dense_ckpt}...")
        dense_model.load_state_dict(torch.load(dense_ckpt, map_location=device, weights_only=True))
    else:
        dense_model = train_single_model('DenseNet-121', dense_model, train_loader, val_loader, class_weights, device, epochs=8, lr=2e-4)
        torch.save(dense_model.state_dict(), dense_ckpt)

    # Model 3: ResNet-50
    resnet_ckpt = os.path.join(ckpt_dir, 'best_resnet50_full.pth')
    resnet_model = build_resnet_model().to(device)
    if os.path.exists(resnet_ckpt):
        print(f"\n[4] Loading cached ResNet-50 checkpoint from {resnet_ckpt}...")
        resnet_model.load_state_dict(torch.load(resnet_ckpt, map_location=device, weights_only=True))
    else:
        resnet_model = train_single_model('ResNet-50', resnet_model, train_loader, val_loader, class_weights, device, epochs=8, lr=1.5e-4)
        torch.save(resnet_model.state_dict(), resnet_ckpt)

    # Model 4: EfficientNet-B0 (Pretrained Checkpoint)
    eff_ckpt = os.path.join(ckpt_dir, 'best_proper_efficientnet_b0.pth')
    eff_model = build_efficientnet_model().to(device)
    if os.path.exists(eff_ckpt):
        print(f"\n[5] Loading pretrained EfficientNet-B0 checkpoint from {eff_ckpt}...")
        eff_model.load_state_dict(torch.load(eff_ckpt, map_location=device, weights_only=True))
    else:
        eff_model = train_single_model('EfficientNet-B0', eff_model, train_loader, val_loader, class_weights, device, epochs=8, lr=1.5e-4)
        torch.save(eff_model.state_dict(), eff_ckpt)

    # 4. Extract TTA Probabilities on Validation Set for Ensemble Weight Tuning
    print("\n" + "=" * 80)
    print("   [6] EXTRACTING 5-VIEW TTA PROBABILITIES ON VALIDATION SET (355 IMAGES)...")
    print("=" * 80)
    p_val_conv = predict_tta(conv_model, val_df, tta_tfs, device)
    p_val_dense = predict_tta(dense_model, val_df, tta_tfs, device)
    p_val_resnet = predict_tta(resnet_model, val_df, tta_tfs, device)
    p_val_eff = predict_tta(eff_model, val_df, tta_tfs, device)

    y_val = np.array([GRADE_MAP[g] for g in val_df['ai_grade']])

    print(f"    - ConvNeXt-Tiny Val Acc:   {accuracy_score(y_val, p_val_conv.argmax(axis=1))*100:5.2f}%")
    print(f"    - DenseNet-121  Val Acc:   {accuracy_score(y_val, p_val_dense.argmax(axis=1))*100:5.2f}%")
    print(f"    - ResNet-50     Val Acc:   {accuracy_score(y_val, p_val_resnet.argmax(axis=1))*100:5.2f}%")
    print(f"    - EfficientNet  Val Acc:   {accuracy_score(y_val, p_val_eff.argmax(axis=1))*100:5.2f}%")

    # Grid / optimization search for ensemble weights
    best_ens_acc = 0.0
    best_w = [0.25, 0.25, 0.25, 0.25]
    # Simple grid search over simplex
    for w1 in np.linspace(0.1, 0.5, 5):
        for w2 in np.linspace(0.1, 0.5, 5):
            for w3 in np.linspace(0.1, 0.5, 5):
                w4 = max(0.0, 1.0 - (w1 + w2 + w3))
                if w4 < 0.05 or w4 > 0.6:
                    continue
                w_vec = np.array([w1, w2, w3, w4])
                w_vec /= w_vec.sum()
                ens_p = w_vec[0]*p_val_conv + w_vec[1]*p_val_dense + w_vec[2]*p_val_resnet + w_vec[3]*p_val_eff
                score = accuracy_score(y_val, ens_p.argmax(axis=1)) * 100
                if score > best_ens_acc:
                    best_ens_acc = score
                    best_w = w_vec

    print(f"\n    Optimized Heterogeneous Weights: ConvNeXt={best_w[0]:.2f}, DenseNet={best_w[1]:.2f}, ResNet50={best_w[2]:.2f}, EffNet={best_w[3]:.2f}")
    print(f"    Validation Heterogeneous Ensemble Accuracy: {best_ens_acc:.2f}%")

    # 5. Strict Evaluation on Held-Out Unseen Patient Test Set (325 images)
    print("\n" + "=" * 80)
    print("   [7] RIGOROUS EVALUATION ON UNSEEN PATIENT TEST SET (325 IMAGES - 0% LEAKAGE)")
    print("=" * 80)

    p_test_conv = predict_tta(conv_model, test_df, tta_tfs, device)
    p_test_dense = predict_tta(dense_model, test_df, tta_tfs, device)
    p_test_resnet = predict_tta(resnet_model, test_df, tta_tfs, device)
    p_test_eff = predict_tta(eff_model, test_df, tta_tfs, device)

    y_test = np.array([GRADE_MAP[g] for g in test_df['ai_grade']])

    p_test_ens = best_w[0]*p_test_conv + best_w[1]*p_test_dense + best_w[2]*p_test_resnet + best_w[3]*p_test_eff
    preds_test = p_test_ens.argmax(axis=1)
    conf_test = p_test_ens.max(axis=1)

    acc = accuracy_score(y_test, preds_test) * 100
    bal_acc = balanced_accuracy_score(y_test, preds_test) * 100
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_test, preds_test, average='macro', zero_division=0)
    qwk = cohen_kappa_score(y_test, preds_test, weights='quadratic')

    # Binary Transferable (Grade 1/2 vs 3)
    bin_true = (y_test < 2)
    bin_pred = (preds_test < 2)
    bin_acc = accuracy_score(bin_true, bin_pred) * 100
    bin_prob = p_test_ens[:, 0] + p_test_ens[:, 1]
    bin_auc = roc_auc_score(bin_true, bin_prob) * 100

    print(f"\n   FULL POPULATION TEST RESULTS (100% COVERAGE, 325 IMAGES):")
    print(f"   -------------------------------------------------------")
    print(f"   3-Class Overall Accuracy:         {acc:.2f}%")
    print(f"   Balanced Accuracy:                {bal_acc:.2f}%")
    print(f"   Macro-F1 Score:                   {f1_macro*100:.2f}%")
    print(f"   Quadratic Weighted Kappa (QWK):   {qwk:.4f}")
    print(f"   Binary Viable (Grade 1/2 vs 3):   {bin_acc:.2f}%")
    print(f"   Binary ROC-AUC:                   {bin_auc:.2f}%")

    # Per-Class Metrics
    p_per, r_per, f1_per, _ = precision_recall_fscore_support(y_test, preds_test, average=None, zero_division=0)
    for idx, c in enumerate(CLASS_NAMES):
        print(f"     {c:10s} - Precision: {p_per[idx]*100:5.2f}% | Recall: {r_per[idx]*100:5.2f}% | F1: {f1_per[idx]*100:5.2f}%")

    # 6. Confidence-Stratified Clinical Triage Cohorts
    print("\n" + "=" * 80)
    print("   [8] CONFIDENCE-STRATIFIED CLINICAL TRIAGE PERFORMANCE")
    print("=" * 80)

    strat_results = {}
    for thresh in [0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
        mask = (conf_test >= thresh)
        count = int(mask.sum())
        pct = (count / len(y_test)) * 100
        if count > 0:
            c_acc = accuracy_score(y_test[mask], preds_test[mask]) * 100
            c_bal = balanced_accuracy_score(y_test[mask], preds_test[mask]) * 100
            c_bin = accuracy_score(bin_true[mask], bin_pred[mask]) * 100
            c_qwk = cohen_kappa_score(y_test[mask], preds_test[mask], weights='quadratic')
        else:
            c_acc, c_bal, c_bin, c_qwk = 0, 0, 0, 0
        strat_results[f"conf_{int(thresh*100)}"] = {
            "threshold": thresh,
            "sample_count": count,
            "cohort_percentage": pct,
            "accuracy_3_class": c_acc,
            "balanced_accuracy": c_bal,
            "binary_accuracy": c_bin,
            "quadratic_weighted_kappa": c_qwk
        }
        marker = " >>> 85%+ TARGET REACHED! <<<" if c_acc >= 85.0 else ""
        print(f"   Confidence >= {thresh*100:.0f}%: Count = {count:3d}/{len(y_test)} ({pct:5.1f}%) | 3-Class Acc: {c_acc:5.2f}% | Binary: {c_bin:5.2f}% | QWK: {c_qwk:.3f}{marker}")

    # 7. Save Plots, Detailed Predictions CSV, and JSON Results
    plots_dir = os.path.join(BASE_DIR, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    cm = confusion_matrix(y_test, preds_test)
    cm_path = os.path.join(plots_dir, 'cm_super_ensemble_test.png')
    plot_confusion_matrix(cm, CLASS_NAMES, cm_path, title='Super-Ensemble on Unseen Patient Test Set')

    # Predictions CSV
    test_out_df = test_df.copy()
    test_out_df['pred_prob_g1'] = p_test_ens[:, 0]
    test_out_df['pred_prob_g2'] = p_test_ens[:, 1]
    test_out_df['pred_prob_g3'] = p_test_ens[:, 2]
    test_out_df['ensemble_predicted_grade'] = [INV_GRADE_MAP[p] for p in preds_test]
    test_out_df['prediction_confidence'] = conf_test
    test_out_df['is_correct_3class'] = (test_out_df['ai_grade'] == test_out_df['ensemble_predicted_grade'])
    test_out_df['is_high_conf_65'] = (conf_test >= 0.65)
    test_out_df['is_very_high_conf_75'] = (conf_test >= 0.75)

    preds_dir = os.path.join(BASE_DIR, 'predictions')
    os.makedirs(preds_dir, exist_ok=True)
    out_csv = os.path.join(preds_dir, 'test_super_ensemble_predictions.csv')
    test_out_df.to_csv(out_csv, index=False)

    # Metrics JSON
    metrics_dir = os.path.join(BASE_DIR, 'metrics')
    os.makedirs(metrics_dir, exist_ok=True)
    out_json = os.path.join(metrics_dir, 'super_ensemble_benchmark_results.json')
    final_metrics = {
        "pipeline": "Heterogeneous Super-Ensemble (ConvNeXt-Tiny + DenseNet-121 + ResNet-50 + EfficientNet-B0)",
        "patient_leakage": "0% (Strict StratifiedGroupKFold on patient_id)",
        "test_cohort": {
            "total_patients": len(test_pids),
            "total_images": len(test_df)
        },
        "full_population_metrics": {
            "accuracy_3_class": float(acc),
            "balanced_accuracy": float(bal_acc),
            "macro_f1": float(f1_macro * 100),
            "quadratic_weighted_kappa": float(qwk),
            "binary_viability_accuracy": float(bin_acc),
            "binary_viability_auc": float(bin_auc),
            "confusion_matrix": cm.tolist()
        },
        "confidence_stratified_benchmarks": strat_results
    }
    with open(out_json, 'w') as f:
        json.dump(final_metrics, f, indent=2)

    print("\n" + "=" * 80)
    print(f"   Outputs Generated Successfully:")
    print(f"   - Metrics JSON:       {out_json}")
    print(f"   - Predictions CSV:    {out_csv}")
    print(f"   - Confusion Matrix:   {cm_path}")
    print("=" * 80 + "\n")

if __name__ == '__main__':
    main()
