"""
Model Ensembling Suite for Embryo Quality Grading
=================================================
Combines out-of-fold probabilistic predictions from the top-performing
diverse architectures:
  - ResNet-50 (with CLAHE Preprocessing)
  - EfficientNet-B0 (Standard RGB, compound scaled)
Evaluates joint agreement, Quadratic Weighted Kappa, and Multiclass ROC-AUC.
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_recall_fscore_support,
    cohen_kappa_score, roc_auc_score, confusion_matrix
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from dataset import CLASS_NAMES, INV_GRADE_MAP

def run_ensemble():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    pred_dir = os.path.join(base_dir, "predictions")
    plot_dir = os.path.join(base_dir, "plots")
    metric_dir = os.path.join(base_dir, "metrics")
    summary_csv = os.path.join(base_dir, "experiments_summary.csv")

    df_r50_clahe = pd.read_csv(os.path.join(pred_dir, "oof_exp3_resnet50_clahe.csv"))
    df_effnet = pd.read_csv(os.path.join(pred_dir, "oof_exp4_effnet_b0.csv"))

    y_true = df_r50_clahe['true_label'].values
    p_r50 = df_r50_clahe[['prob_grade1', 'prob_grade2', 'prob_grade3']].values.astype(np.float64)
    p_eff = df_effnet[['prob_grade1', 'prob_grade2', 'prob_grade3']].values.astype(np.float64)

    # Soft average ensemble
    p_ens = 0.5 * p_r50 + 0.5 * p_eff
    p_ens = p_ens / p_ens.sum(axis=1, keepdims=True)
    preds = p_ens.argmax(axis=1)

    acc = accuracy_score(y_true, preds)
    bal_acc = balanced_accuracy_score(y_true, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, preds, average='macro')
    qwk = cohen_kappa_score(y_true, preds, weights='quadratic')
    auc = roc_auc_score(y_true, p_ens, multi_class='ovr', average='macro')

    prec_pc, rec_pc, f1_pc, supp_pc = precision_recall_fscore_support(y_true, preds, average=None)

    oof_ens = df_r50_clahe[['filename', 'fold', 'true_label', 'true_grade']].copy()
    oof_ens['pred_label'] = preds
    oof_ens['pred_grade'] = oof_ens['pred_label'].map(INV_GRADE_MAP)
    oof_ens['prob_grade1'] = p_ens[:, 0]
    oof_ens['prob_grade2'] = p_ens[:, 1]
    oof_ens['prob_grade3'] = p_ens[:, 2]
    oof_ens.to_csv(os.path.join(pred_dir, "oof_exp6_ensemble_r50_effnet.csv"), index=False)

    cm = confusion_matrix(y_true, preds)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm_norm, cmap=plt.cm.Greens, vmin=0, vmax=1.0)
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
    ax.set_title(f"Confusion Matrix: exp6_ensemble_r50_effnet\n(OOF Acc: {acc*100:.2f}%, QWK: {qwk:.4f}, AUC: {auc*100:.2f}%)", fontweight='bold', pad=10)
    ax.set_xlabel('Predicted Grade', fontweight='bold')
    ax.set_ylabel('True Grade', fontweight='bold')
    plt.tight_layout()
    cm_path = os.path.join(plot_dir, "cm_exp6_ensemble_r50_effnet.png")
    plt.savefig(cm_path, dpi=200)
    plt.close()

    metrics = {
        'exp_id': 'exp6_ensemble_r50_effnet',
        'arch_name': 'ensemble (resnet50_clahe + efficientnet_b0)',
        'img_size': 224,
        'preprocessing': 'Hybrid (CLAHE + Standard RGB)',
        'loss_type': 'cross_entropy',
        'overall_accuracy': float(acc),
        'balanced_accuracy': float(bal_acc),
        'macro_precision': float(prec),
        'macro_recall': float(rec),
        'macro_f1': float(f1),
        'quadratic_weighted_kappa': float(qwk),
        'macro_roc_auc': float(auc),
        'class_metrics': {
            'Grade-1': {'precision': float(prec_pc[0]), 'recall': float(rec_pc[0]), 'f1': float(f1_pc[0]), 'support': int(supp_pc[0])},
            'Grade-2': {'precision': float(prec_pc[1]), 'recall': float(rec_pc[1]), 'f1': float(f1_pc[1]), 'support': int(supp_pc[1])},
            'Grade-3': {'precision': float(prec_pc[2]), 'recall': float(rec_pc[2]), 'f1': float(f1_pc[2]), 'support': int(supp_pc[2])}
        },
        'confusion_matrix': cm.tolist()
    }

    with open(os.path.join(metric_dir, "exp6_ensemble_r50_effnet_metrics.json"), 'w') as f:
        json.dump(metrics, f, indent=2)

    row = {
        'Experiment_ID': 'exp6_ensemble_r50_effnet',
        'Architecture': 'ensemble(r50_clahe+effnet_b0)',
        'Resolution': '224x224',
        'Preprocessing': 'Hybrid (CLAHE+RGB)',
        'Loss_Function': 'cross_entropy',
        'Probe_LR': 'N/A',
        'FineTune_LR': 'N/A',
        'Batch_Size': 16,
        'Total_Epochs': 'N/A',
        'Fold_Accuracies': 'N/A',
        'CV_Mean_Acc': round(acc * 100, 2),
        'CV_Std_Acc': 0.0,
        'Overall_OOF_Acc': round(acc * 100, 2),
        'Balanced_Acc': round(bal_acc * 100, 2),
        'Macro_F1': round(f1 * 100, 2),
        'Cohen_QWK': round(qwk, 4),
        'Macro_ROC_AUC': round(auc * 100, 2),
        'Duration_Sec': 0.0,
        'GPU_Error': 'None'
    }

    if os.path.exists(summary_csv):
        df_sum = pd.read_csv(summary_csv)
        # Avoid duplicate row
        df_sum = df_sum[df_sum['Experiment_ID'] != 'exp6_ensemble_r50_effnet']
        df_sum = pd.concat([df_sum, pd.DataFrame([row])], ignore_index=True)
        df_sum.to_csv(summary_csv, index=False)

    print("Ensemble evaluation complete!")
    print(f"OOF Acc: {acc*100:.2f}% | Balanced Acc: {bal_acc*100:.2f}% | Macro F1: {f1*100:.2f}% | QWK: {qwk:.4f} | ROC-AUC: {auc*100:.2f}%")

if __name__ == '__main__':
    run_ensemble()
