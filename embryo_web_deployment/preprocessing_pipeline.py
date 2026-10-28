"""
Embryo Image Preprocessing Pipeline
===================================
Production-ready preprocessing module for Embryo Grading Web Application.

Functions:
  1. compute_sharpness(image):
       Calculates optical focus score via Laplacian gradient variance.
       Use this when a user uploads multiple Z-stack slices to automatically pick the best in-focus slice!
  2. preprocess_embryo_image(image_input, target_size=256):
       - Square Center Crop (focusing on the blastocyst and removing peripheral vignetting)
       - High-quality Lanczos interpolation to target size (256x256)
       - CLAHE (Contrast-Limited Adaptive Histogram Equalization) in LAB space to enhance Inner Cell Mass (ICM)
         and trophectoderm cell borders.
  3. get_tensor_transform(target_size=256):
       Returns torchvision transforms to convert the preprocessed image into model-ready tensors.
"""

import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms

def compute_sharpness(image_input):
    """
    Computes optical focus sharpness using the Variance of the Laplacian.
    Args:
        image_input: File path (str), NumPy array (BGR/Grayscale), or PIL Image.
    Returns:
        float: Sharpness score (higher = sharper and more in-focus).
    """
    if isinstance(image_input, str):
        img_gray = cv2.imread(image_input, cv2.IMREAD_GRAYSCALE)
    elif isinstance(image_input, Image.Image):
        img_np = np.array(image_input)
        if len(img_np.shape) == 3:
            img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        else:
            img_gray = img_np
    elif isinstance(image_input, np.ndarray):
        if len(image_input.shape) == 3:
            img_gray = cv2.cvtColor(image_input, cv2.COLOR_BGR2GRAY)
        else:
            img_gray = image_input
    else:
        raise ValueError("Unsupported image input type.")

    if img_gray is None:
        return 0.0

    return float(cv2.Laplacian(img_gray, cv2.CV_64F).var())

def select_best_slice(image_paths):
    """
    Given a list of slice image paths for a patient (Z-stack), returns the sharpest in-focus slice.
    """
    if not image_paths:
        return None
    scored = [(path, compute_sharpness(path)) for path in image_paths]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[0][0]

def preprocess_embryo_image(image_input, target_size=256, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Preprocesses an embryo image with square aspect cropping and LAB CLAHE enhancement.
    Args:
        image_input: File path (str), NumPy BGR array, or PIL Image.
        target_size: Int, output spatial resolution (default 256).
    Returns:
        PIL.Image: Preprocessed RGB image ready for display or model input.
    """
    # 1. Load image as BGR numpy array
    if isinstance(image_input, str):
        img_bgr = cv2.imread(image_input)
        if img_bgr is None:
            raise FileNotFoundError(f"Could not read image from path: {image_input}")
    elif isinstance(image_input, Image.Image):
        img_rgb = np.array(image_input.convert('RGB'))
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    elif isinstance(image_input, np.ndarray):
        img_bgr = image_input.copy()
    else:
        raise ValueError("Unsupported image input type.")

    # 2. Square Center Crop (removes peripheral camera vignetting and dark margins)
    h, w = img_bgr.shape[:2]
    sz = min(h, w)
    y_start = (h - sz) // 2
    x_start = (w - sz) // 2
    cropped = img_bgr[y_start:y_start+sz, x_start:x_start+sz]

    # 3. High-Quality Resize
    resized = cv2.resize(cropped, (target_size, target_size), interpolation=cv2.INTER_LANCZOS4)

    # 4. CLAHE Contrast Enhancement in LAB color space
    lab = cv2.cvtColor(resized, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    enhanced_bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # Convert to RGB PIL Image
    enhanced_rgb = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(enhanced_rgb)

def get_tensor_transform(target_size=256):
    """
    Returns standard torchvision transform for normalizing image tensors to ImageNet statistics.
    """
    return transforms.Compose([
        transforms.Resize((target_size, target_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

if __name__ == '__main__':
    import os
    print("Embryo Preprocessing Pipeline Module ready.")
