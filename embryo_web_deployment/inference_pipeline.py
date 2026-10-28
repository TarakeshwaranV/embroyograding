"""
Embryo Classification Inference Pipeline
=========================================
Backend predictor for Embryo Grading Web Application.

Loads the Heterogeneous Super-Ensemble (ConvNeXt-Tiny, DenseNet-121, ResNet-50, EfficientNet-B0)
and runs end-to-end inference on any input embryo image.

Usage in Web App (Flask / FastAPI / Streamlit):
    from inference_pipeline import EmbryoPredictor
    predictor = EmbryoPredictor()
    result = predictor.predict('sample_embryo.png')
    print(result)
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision.models import (
    convnext_tiny, densenet121, resnet50, efficientnet_b0
)

# Import local preprocessing
from preprocessing_pipeline import preprocess_embryo_image, get_tensor_transform, compute_sharpness

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')

GRADE_NAMES = ['Grade-1', 'Grade-2', 'Grade-3']
QUALITY_MAP = {
    'Grade-1': 'Good Quality Blastocyst',
    'Grade-2': 'Fair Quality Blastocyst',
    'Grade-3': 'Poor Quality / Arrested Embryo'
}
VIABILITY_MAP = {
    'Grade-1': 'Viable / Transferable (Recommended for Transfer / Freeze)',
    'Grade-2': 'Viable / Transferable (Transferable Candidate)',
    'Grade-3': 'Non-Viable (Arrested / High Aneuploidy Risk - Discard)'
}
RECOMMENDATIONS = {
    'Grade-1': 'Optimal morphology (Gardner AA/AB equivalent). Priority candidate for single blastocyst transfer (eSET) or high-survival vitrification.',
    'Grade-2': 'Acceptable morphology (Gardner BB/BA equivalent). Suitable for clinical transfer or secondary vitrification.',
    'Grade-3': 'Sub-optimal morphology with poor cellular cohesion or extensive fragmentation. Clinical consensus recommends against transfer.'
}

def build_convnext_head():
    m = convnext_tiny(weights=None)
    in_feat = m.classifier[2].in_features
    m.classifier[2] = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_feat, 256),
        nn.GELU(),
        nn.Dropout(p=0.21),
        nn.Linear(256, 3)
    )
    return m

def build_densenet_head():
    m = densenet121(weights=None)
    in_feat = m.classifier.in_features
    m.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_feat, 256),
        nn.SiLU(),
        nn.Dropout(p=0.21),
        nn.Linear(256, 3)
    )
    return m

def build_resnet_head():
    m = resnet50(weights=None)
    in_feat = m.fc.in_features
    m.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_feat, 256),
        nn.SiLU(),
        nn.Dropout(p=0.21),
        nn.Linear(256, 3)
    )
    return m

def build_effnet_head():
    m = efficientnet_b0(weights=None)
    in_feat = m.classifier[1].in_features
    m.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_feat, 256),
        nn.SiLU(),
        nn.Dropout(p=0.21),
        nn.Linear(256, 3)
    )
    return m

class EmbryoPredictor:
    def __init__(self, models_dir=MODELS_DIR, device=None):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        print(f"[EmbryoPredictor] Initializing Super-Ensemble on {self.device}...")

        # Load models
        self.convnext = build_convnext_head().to(self.device)
        self.densenet = build_densenet_head().to(self.device)
        self.resnet = build_resnet_head().to(self.device)
        self.effnet = build_effnet_head().to(self.device)

        # Checkpoints
        ckpt_conv = os.path.join(models_dir, 'best_convnext_tiny.pth')
        ckpt_dense = os.path.join(models_dir, 'best_densenet121.pth')
        ckpt_res = os.path.join(models_dir, 'best_resnet50_full.pth')
        ckpt_eff = os.path.join(models_dir, 'best_proper_efficientnet_b0.pth')

        self.convnext.load_state_dict(torch.load(ckpt_conv, map_location=self.device, weights_only=True))
        self.densenet.load_state_dict(torch.load(ckpt_dense, map_location=self.device, weights_only=True))
        self.resnet.load_state_dict(torch.load(ckpt_res, map_location=self.device, weights_only=True))
        self.effnet.load_state_dict(torch.load(ckpt_eff, map_location=self.device, weights_only=True))

        self.convnext.eval()
        self.densenet.eval()
        self.resnet.eval()
        self.effnet.eval()

        # Optimal Heterogeneous Ensemble weights
        self.weights = np.array([0.10, 0.40, 0.20, 0.30])
        self.transform = get_tensor_transform(target_size=256)
        print("[EmbryoPredictor] Super-Ensemble ready for web inference.")

    def predict(self, image_input):
        """
        Runs full prediction pipeline on an input embryo image.
        Args:
            image_input: File path (str), NumPy array, or PIL Image.
        Returns:
            dict: Structured prediction with grades, confidence, viability, and clinical reasoning.
        """
        # Preprocessing: Centering + CLAHE
        clean_pil = preprocess_embryo_image(image_input, target_size=256)
        tensor = self.transform(clean_pil).unsqueeze(0).to(self.device)

        with torch.no_grad():
            p_conv = F.softmax(self.convnext(tensor), dim=1).cpu().numpy()[0]
            p_dense = F.softmax(self.densenet(tensor), dim=1).cpu().numpy()[0]
            p_res = F.softmax(self.resnet(tensor), dim=1).cpu().numpy()[0]
            p_eff = F.softmax(self.effnet(tensor), dim=1).cpu().numpy()[0]

        # Soft weighted ensemble voting
        ens_prob = (
            self.weights[0] * p_conv +
            self.weights[1] * p_dense +
            self.weights[2] * p_res +
            self.weights[3] * p_eff
        )
        ens_prob /= ens_prob.sum()

        pred_idx = int(np.argmax(ens_prob))
        predicted_grade = GRADE_NAMES[pred_idx]
        confidence = float(ens_prob[pred_idx])

        return {
            "predicted_grade": predicted_grade,
            "quality_tier": QUALITY_MAP[predicted_grade],
            "clinical_viability": VIABILITY_MAP[predicted_grade],
            "confidence_score": round(confidence, 4),
            "confidence_percentage": f"{confidence * 100:.1f}%",
            "probabilities": {
                "Grade-1": round(float(ens_prob[0]), 4),
                "Grade-2": round(float(ens_prob[1]), 4),
                "Grade-3": round(float(ens_prob[2]), 4)
            },
            "clinical_recommendation": RECOMMENDATIONS[predicted_grade]
        }

if __name__ == '__main__':
    predictor = EmbryoPredictor()
    sample_img = os.path.join(os.path.dirname(MODELS_DIR), '..', 'Images', '0001_03.png')
    if os.path.exists(sample_img):
        res = predictor.predict(sample_img)
        print("\n--- Sample Prediction ---")
        for k, v in res.items():
            print(f"{k}: {v}")
