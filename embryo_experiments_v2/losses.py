"""
Loss Functions for Embryo Quality Grading
=========================================
Implements:
  1. Standard Class-Weighted Cross-Entropy Loss
  2. Ordinal-Aware Soft Cross-Entropy Loss (penalizes distance-2 grade errors)
  3. Focal Cross-Entropy Loss (focuses on hard borderline blastocysts)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class OrdinalSoftLoss(nn.Module):
    """
    Ordinal classification loss using distance-proportional soft targets.
    In embryo grading:
      - Grade-1 (Good) vs Grade-2 (Fair): distance = 1 (small biological discrepancy)
      - Grade-1 (Good) vs Grade-3 (Poor): distance = 2 (severe catastrophic error)
    Target distribution matrix (smoothing=0.15):
      Grade-1 -> [0.85, 0.15, 0.00]
      Grade-2 -> [0.075, 0.85, 0.075]
      Grade-3 -> [0.00, 0.15, 0.85]
    """
    def __init__(self, smoothing=0.15):
        super(OrdinalSoftLoss, self).__init__()
        self.smoothing = smoothing
        self.soft_matrix = torch.tensor([
            [1.0 - smoothing, smoothing, 0.0],
            [smoothing / 2.0, 1.0 - smoothing, smoothing / 2.0],
            [0.0, smoothing, 1.0 - smoothing]
        ], dtype=torch.float32)

    def forward(self, logits, targets):
        log_probs = F.log_softmax(logits, dim=1)
        matrix = self.soft_matrix.to(logits.device)
        soft_targets = matrix[targets]
        loss = -(soft_targets * log_probs).sum(dim=1).mean()
        return loss

class FocalLoss(nn.Module):
    """Focal Loss down-weights well-classified examples to focus on difficult intermediate blastocysts."""
    def __init__(self, alpha=None, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()

def get_loss_criterion(loss_type='cross_entropy', class_weights=None, smoothing=0.15):
    if loss_type == 'cross_entropy':
        return nn.CrossEntropyLoss(weight=class_weights)
    elif loss_type == 'ordinal':
        return OrdinalSoftLoss(smoothing=smoothing)
    elif loss_type == 'focal':
        return FocalLoss(alpha=class_weights, gamma=2.0)
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")
