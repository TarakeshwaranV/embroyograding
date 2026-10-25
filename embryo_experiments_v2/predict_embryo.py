"""
Stand-Alone Inference Script for Embryo Quality Grading
======================================================
Classifies an input blastocyst microscopy image into:
  - Grade 1: Good quality
  - Grade 2: Fair quality
  - Grade 3: Poor quality

Outputs:
  - Predicted grade
  - Uncalibrated model probability estimates for all three classes
  - Diagnostic warning: probabilities are model confidence estimates, not clinical diagnosis.
"""

import os
import sys
import argparse
import numpy as np
from PIL import Image

import torch
import torchvision.transforms as transforms

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models import get_model
from dataset import get_transforms, CLASS_NAMES, INV_GRADE_MAP

def predict_single_image(image_path, model_path='embryo_experiments_v2/checkpoints/best_exp3_resnet50_clahe.pth', arch_name='resnet50', img_size=224, use_clahe=False):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Embryo image not found: {image_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}. Train the model first.")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load Model
    model = get_model(arch_name=arch_name, num_classes=3, pretrained=False)
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    # Preprocessing
    image = Image.open(image_path).convert('RGB')
    if use_clahe or 'clahe' in model_path.lower():
        from dataset import CLAHETransform
        clahe = CLAHETransform()
        image = clahe(image)

    transform = get_transforms(img_size=img_size, is_training=False)
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred_idx = probs.argmax()

    pred_grade = INV_GRADE_MAP[pred_idx]

    result = {
        'image_path': os.path.abspath(image_path),
        'predicted_grade': pred_grade,
        'model_architecture': arch_name,
        'checkpoint_used': os.path.abspath(model_path),
        'class_probabilities': {
            'Grade-1 (Good)': float(probs[0]),
            'Grade-2 (Fair)': float(probs[1]),
            'Grade-3 (Poor)': float(probs[2])
        },
        'clinical_notice': 'PROBABILITY ESTIMATES ARE EXPERIMENTAL AI PREDICTIONS, NOT CLINICALLY CALIBRATED DIAGNOSTIC METRICS.'
    }

    return result

def predict_ensemble(image_path, r50_ckpt='embryo_experiments_v2/checkpoints/best_exp3_resnet50_clahe.pth', eff_ckpt='embryo_experiments_v2/checkpoints/best_exp4_effnet_b0.pth'):
    res_r50 = predict_single_image(image_path, model_path=r50_ckpt, arch_name='resnet50', use_clahe=True)
    res_eff = predict_single_image(image_path, model_path=eff_ckpt, arch_name='efficientnet_b0', use_clahe=False)

    p_r50 = np.array([res_r50['class_probabilities']['Grade-1 (Good)'], res_r50['class_probabilities']['Grade-2 (Fair)'], res_r50['class_probabilities']['Grade-3 (Poor)']])
    p_eff = np.array([res_eff['class_probabilities']['Grade-1 (Good)'], res_eff['class_probabilities']['Grade-2 (Fair)'], res_eff['class_probabilities']['Grade-3 (Poor)']])

    p_ens = 0.5 * p_r50 + 0.5 * p_eff
    p_ens = p_ens / p_ens.sum()
    pred_idx = p_ens.argmax()
    pred_grade = INV_GRADE_MAP[pred_idx]

    result = {
        'image_path': os.path.abspath(image_path),
        'predicted_grade': pred_grade,
        'model_architecture': 'Ensemble: ResNet-50 (CLAHE) + EfficientNet-B0',
        'checkpoint_used': f"{os.path.basename(r50_ckpt)} + {os.path.basename(eff_ckpt)}",
        'class_probabilities': {
            'Grade-1 (Good)': float(p_ens[0]),
            'Grade-2 (Fair)': float(p_ens[1]),
            'Grade-3 (Poor)': float(p_ens[2])
        },
        'clinical_notice': 'PROBABILITY ESTIMATES ARE EXPERIMENTAL AI PREDICTIONS, NOT CLINICALLY CALIBRATED DIAGNOSTIC METRICS.'
    }
    return result

def main():
    parser = argparse.ArgumentParser(description="Classify Blastocyst Embryo Image")
    parser.add_argument('--image', type=str, required=True, help="Path to embryo image file (e.g. Images/0002_01.png)")
    parser.add_argument('--model', type=str, default='embryo_experiments_v2/checkpoints/best_exp3_resnet50_clahe.pth', help="Path to checkpoint .pth file")
    parser.add_argument('--arch', type=str, default='resnet50', choices=['resnet50', 'efficientnet_b0', 'resnet18'])
    parser.add_argument('--ensemble', action='store_true', help="Use soft ensemble of ResNet-50 (CLAHE) + EfficientNet-B0")
    args = parser.parse_args()

    if args.ensemble:
        res = predict_ensemble(args.image)
    else:
        res = predict_single_image(args.image, model_path=args.model, arch_name=args.arch)

    print("\n" + "="*60)
    print("      HUMAN BLASTOCYST QUALITY GRADING RESULT")
    print("="*60)
    print(f" Image:             {os.path.basename(res['image_path'])}")
    print(f" Model Used:        {res['model_architecture'].upper()}")
    print(f" Checkpoint:        {os.path.basename(res['checkpoint_used'])}")
    print("-" * 60)
    print(f" PREDICTED GRADE:   >>> {res['predicted_grade']} <<<")
    print("-" * 60)
    print(" Estimated Class Probabilities:")
    for grade_name, prob in res['class_probabilities'].items():
        bar = "#" * int(prob * 30)
        print(f"   {grade_name:18s}: {prob*100:6.2f}% | {bar}")
    print("="*60)
    print(f" [!] Notice: {res['clinical_notice']}")
    print("="*60 + "\n")

if __name__ == '__main__':
    main()
