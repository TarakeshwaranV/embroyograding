"""
Diagnostic Grad-CAM Visualization for Human Blastocyst Grading
==============================================================
Generates class activation heatmaps to inspect whether the network attends
to true biological landmarks (Inner Cell Mass, Trophectoderm, Zona Pellucida)
or extraneous artifacts (illumination gradients, incubator edges).
"""

import os
import sys
import cv2
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from dataset import get_transforms, CLASS_NAMES, INV_GRADE_MAP
from models import get_model

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor, target_class=None):
        self.model.eval()
        self.model.zero_grad()

        output = self.model(input_tensor)
        if target_class is None:
            target_class = output.argmax(dim=1).item()

        score = output[0, target_class]
        score.backward()

        # Global average pooling of gradients
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        cam = cam.squeeze().cpu().numpy()
        cam = cv2.resize(cam, (input_tensor.shape[3], input_tensor.shape[2]))
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam, target_class, output

def overlay_heatmap(image_pil, cam, alpha=0.5):
    img_np = np.array(image_pil)
    h, w, _ = img_np.shape
    cam_resized = cv2.resize(cam, (w, h))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlaid = np.uint8(alpha * heatmap + (1 - alpha) * img_np)
    return overlaid

def run_gradcam_diagnostics(model_path, arch_name='resnet50', sample_images=None, out_dir='embryo_experiments_v2/plots/gradcam'):
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = get_model(arch_name=arch_name, num_classes=3, pretrained=False)
    state_dict = torch.load(model_path, map_location='cpu', weights_only=True)
    model.load_state_dict(state_dict)
    model = model.to(device)

    # Select target convolutional layer
    if 'resnet' in arch_name:
        target_layer = model.layer4[-1]
    elif 'efficientnet' in arch_name:
        target_layer = model.features[-1]
    else:
        raise ValueError(f"Unknown architecture: {arch_name}")

    gradcam = GradCAM(model, target_layer)
    transform = get_transforms(img_size=224, is_training=False)

    if sample_images is None:
        sample_images = [
            ('Images/0002_01.png', 'Grade-2'),
            ('Images/0005_01.png', 'Grade-1'),
            ('Images/0010_01.png', 'Grade-3')
        ]

    for img_path, true_grade in sample_images:
        if not os.path.exists(img_path):
            continue
        orig_img = Image.open(img_path).convert('RGB')
        input_tensor = transform(orig_img).unsqueeze(0).to(device)

        cam, pred_class, logits = gradcam.generate(input_tensor)
        probs = torch.softmax(logits, dim=1).detach().cpu().numpy()[0]
        pred_grade = INV_GRADE_MAP[pred_class]

        overlaid = overlay_heatmap(orig_img, cam, alpha=0.45)
        fname = os.path.basename(img_path).split('.')[0]
        save_path = os.path.join(out_dir, f"gradcam_{fname}_true_{true_grade}_pred_{pred_grade}.png")
        Image.fromarray(overlaid).save(save_path)
        print(f"Saved Grad-CAM for {fname}: True={true_grade}, Pred={pred_grade} ({probs[pred_class]*100:.1f}%) -> {save_path}", flush=True)
