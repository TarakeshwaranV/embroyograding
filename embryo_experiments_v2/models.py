"""
Model Architectures and Transfer Learning Suite
================================================
Implements:
  - ResNet-50
  - EfficientNet-B0
  - ResNet-18
with two-stage transfer learning (linear probing -> differential fine-tuning).
"""

import os
import torch
import torch.nn as nn
import torchvision.models as models

def get_model(arch_name='resnet50', num_classes=3, pretrained=True, dropout_rate=0.3):
    cache_dir = os.path.expanduser(r"~\.cache\torch\hub\checkpoints")

    if arch_name == 'resnet50':
        model = models.resnet50(weights=None)
        if pretrained:
            p = os.path.join(cache_dir, "resnet50-11ad3fa6.pth")
            if os.path.exists(p):
                model.load_state_dict(torch.load(p, weights_only=True))
            else:
                model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, num_classes)
        )

    elif arch_name == 'resnet18':
        model = models.resnet18(weights=None)
        if pretrained:
            p = os.path.join(cache_dir, "resnet18-f37072fd.pth")
            if os.path.exists(p):
                model.load_state_dict(torch.load(p, weights_only=True))
            else:
                model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, num_classes)
        )

    elif arch_name == 'efficientnet_b0':
        model = models.efficientnet_b0(weights=None)
        if pretrained:
            p = os.path.join(cache_dir, "efficientnet_b0_rwightman-7f5810bc.pth")
            if os.path.exists(p):
                model.load_state_dict(torch.load(p, weights_only=True))
            else:
                model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, num_classes)
        )
    else:
        raise ValueError(f"Unsupported architecture: {arch_name}")

    return model

def freeze_feature_extractor(model, arch_name):
    """
    Stage 1: Freeze backbone feature extractors and train ONLY the classification head.
    Prevents large gradient shocks from disturbing pretrained features.
    """
    if 'resnet' in arch_name:
        for name, param in model.named_parameters():
            if 'fc' not in name:
                param.requires_grad = False
            else:
                param.requires_grad = True
    elif 'efficientnet' in arch_name:
        for name, param in model.named_parameters():
            if 'classifier' not in name:
                param.requires_grad = False
            else:
                param.requires_grad = True

def unfreeze_backbone(model, arch_name, unfreeze_all=False):
    """
    Stage 2: Unfreeze higher conv layers (or full backbone) for gentle fine-tuning.
    """
    if unfreeze_all:
        for param in model.parameters():
            param.requires_grad = True
    else:
        if 'resnet' in arch_name:
            for name, param in model.named_parameters():
                # Unfreeze layer3, layer4 and fc
                if any(x in name for x in ['layer3', 'layer4', 'fc']):
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        elif 'efficientnet' in arch_name:
            for name, param in model.named_parameters():
                # Unfreeze top feature blocks (blocks 5, 6, 7) and classifier
                if any(x in name for x in ['features.6', 'features.7', 'features.8', 'classifier']):
                    param.requires_grad = True
                else:
                    param.requires_grad = False
