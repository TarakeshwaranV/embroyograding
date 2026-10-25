"""
Master Experiment Runner
========================
Executes Phase 3, 4, 5, 6, 7 comparison matrix:
  Exp 1: ResNet-50 + Standard Cross-Entropy (Two-Stage)
  Exp 2: ResNet-50 + Ordinal Soft Loss
  Exp 3: ResNet-50 + CLAHE Preprocessing
  Exp 4: EfficientNet-B0 (Two-Stage Transfer Learning)
  Exp 5: ResNet-18 (Compact Architecture for Small Data)
"""

import os
import sys
import argparse

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from train_cv import run_experiment

def main():
    parser = argparse.ArgumentParser(description="Run Structured Embryo Experiments")
    parser.add_argument('--exp', type=str, default='all', choices=['all', 'exp1', 'exp2', 'exp3', 'exp4', 'exp5'])
    parser.add_argument('--force', action='store_true', help="Force re-running already completed experiments")
    args = parser.parse_args()

    default_config = {
        'img_size': 224,
        'batch_size': 16,
        'head_lr': 1e-3,
        'finetune_lr': 1e-4,
        'weight_decay': 1e-4,
        'probe_epochs': 5,
        'finetune_epochs': 15,
        'dropout': 0.3,
        'seed': 42,
        'use_clahe': False,
        'use_class_weights': True,
        'loss_type': 'cross_entropy',
        'ordinal_smoothing': 0.15,
        'unfreeze_all': False
    }

    experiments = {
        'exp1_resnet50_ce': {
            'arch': 'resnet50',
            'config': {**default_config, 'loss_type': 'cross_entropy', 'use_clahe': False}
        },
        'exp2_resnet50_ordinal': {
            'arch': 'resnet50',
            'config': {**default_config, 'loss_type': 'ordinal', 'use_clahe': False}
        },
        'exp3_resnet50_clahe': {
            'arch': 'resnet50',
            'config': {**default_config, 'loss_type': 'cross_entropy', 'use_clahe': True}
        },
        'exp4_effnet_b0': {
            'arch': 'efficientnet_b0',
            'config': {**default_config, 'loss_type': 'cross_entropy', 'use_clahe': False}
        },
        'exp5_resnet18': {
            'arch': 'resnet18',
            'config': {**default_config, 'loss_type': 'cross_entropy', 'use_clahe': False}
        }
    }

    exps_to_run = list(experiments.keys()) if args.exp == 'all' else [k for k in experiments if args.exp in k]

    base_dir = os.path.dirname(os.path.abspath(__file__))
    metrics_dir = os.path.join(base_dir, "metrics")

    for exp_id in exps_to_run:
        metric_file = os.path.join(metrics_dir, f"{exp_id}_metrics.json")
        if os.path.exists(metric_file) and not args.force:
            print(f">>> SKIPPING ALREADY COMPLETED: {exp_id} (use --force to rerun) <<<", flush=True)
            continue

        info = experiments[exp_id]
        print(f"\n>>> LAUNCHING: {exp_id} <<<", flush=True)
        run_experiment(exp_id, info['arch'], info['config'])

    print("\nAll requested experiments finished! Check embryo_experiments_v2/experiments_summary.csv for results.", flush=True)

if __name__ == '__main__':
    main()
