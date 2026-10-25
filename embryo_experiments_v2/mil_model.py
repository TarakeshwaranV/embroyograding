"""
Gated Attention Multi-Instance Learning (MIL) Architecture
==========================================================
Implements Gated Attention Pooling (Ilse et al., ICML 2018) for multi-focal embryo grading.
Each embryo is represented as a bag of focal slices:
  Input: (K, 3, H, W) where K is the number of focal planes
  Output:
    - Logits: (1, 3) predicted class scores
    - Attention Weights: (K,) clinical slice importance weights
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class GatedAttentionMIL(nn.Module):
    def __init__(self, arch_name='resnet50', num_classes=3, pretrained=True, emb_dim=512, attn_dim=128, dropout=0.3):
        super(GatedAttentionMIL, self).__init__()
        self.arch_name = arch_name.lower()
        self.num_classes = num_classes

        # 1. Feature Extractor Backbone
        cache_dir = os.path.expanduser(r"~\.cache\torch\hub\checkpoints")
        if 'resnet50' in self.arch_name:
            backbone = models.resnet50(weights=None)
            if pretrained:
                p = os.path.join(cache_dir, "resnet50-11ad3fa6.pth")
                if os.path.exists(p):
                    backbone.load_state_dict(torch.load(p, weights_only=True))
            in_features = backbone.fc.in_features
            # Remove final fc layer
            self.backbone = nn.Sequential(*list(backbone.children())[:-1], nn.Flatten())
        elif 'efficientnet_b0' in self.arch_name:
            backbone = models.efficientnet_b0(weights=None)
            if pretrained:
                p = os.path.join(cache_dir, "efficientnet_b0_rwightman-7f5810bc.pth")
                if os.path.exists(p):
                    backbone.load_state_dict(torch.load(p, weights_only=True))
            in_features = backbone.classifier[1].in_features
            backbone.classifier = nn.Identity()
            self.backbone = backbone
        elif 'resnet18' in self.arch_name:
            backbone = models.resnet18(weights=None)
            if pretrained:
                p = os.path.join(cache_dir, "resnet18-f37072fd.pth")
                if os.path.exists(p):
                    backbone.load_state_dict(torch.load(p, weights_only=True))
            in_features = backbone.fc.in_features
            self.backbone = nn.Sequential(*list(backbone.children())[:-1], nn.Flatten())
        else:
            raise ValueError(f"Unsupported architecture: {arch_name}")

        # 2. Slice Feature Projection
        self.feature_proj = nn.Sequential(
            nn.Linear(in_features, emb_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # 3. Gated Attention Mechanism
        self.attention_V = nn.Sequential(
            nn.Linear(emb_dim, attn_dim),
            nn.Tanh()
        )
        self.attention_U = nn.Sequential(
            nn.Linear(emb_dim, attn_dim),
            nn.Sigmoid()
        )
        self.attention_weights = nn.Linear(attn_dim, 1)

        # 4. Bag-Level Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(emb_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, bag_tensor):
        """
        bag_tensor: (K, 3, H, W) representing K focal slices of one embryo
        Returns:
          logits: (1, num_classes)
          attention_scores: (K,)
        """
        # Feature extraction for each slice
        features = self.backbone(bag_tensor)  # (K, in_features)
        h = self.feature_proj(features)       # (K, emb_dim)

        # Gated Attention Pooling
        a_v = self.attention_V(h)             # (K, attn_dim)
        a_u = self.attention_U(h)             # (K, attn_dim)
        a_gate = a_v * a_u                    # (K, attn_dim)
        a_scores = self.attention_weights(a_gate) # (K, 1)
        a_norm = F.softmax(a_scores, dim=0)   # (K, 1) - sums to 1.0

        # Aggregated bag embedding: weighted sum across slices
        z_embryo = torch.sum(a_norm * h, dim=0, keepdim=True) # (1, emb_dim)

        # Classify embryo
        logits = self.classifier(z_embryo)    # (1, num_classes)

        return logits, a_norm.squeeze(-1)

def freeze_mil_backbone(model):
    for param in model.backbone.parameters():
        param.requires_grad = False

def unfreeze_mil_backbone(model):
    if 'resnet50' in model.arch_name or 'resnet18' in model.arch_name:
        # Unfreeze layer4 and layer3
        for param in model.backbone.parameters():
            param.requires_grad = True
    elif 'efficientnet' in model.arch_name:
        for param in model.backbone.parameters():
            param.requires_grad = True
