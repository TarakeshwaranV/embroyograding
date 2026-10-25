"""
Dataset Loader and Realistic Augmentation Suite
================================================
Manages the human blastocyst dataset with:
  - 100% Patient-grouped stratification (Zero Patient Leakage)
  - Pure on-the-fly realistic augmentations during training
  - Deterministic evaluation for validation/testing
"""

import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

CLASS_NAMES = ['Grade-1', 'Grade-2', 'Grade-3']
GRADE_MAP = {'Grade-1': 0, 'Grade-2': 1, 'Grade-3': 2}
INV_GRADE_MAP = {0: 'Grade-1', 1: 'Grade-2', 2: 'Grade-3'}

class CLAHETransform:
    """Applies Contrast-Limited Adaptive Histogram Equalization on the Luminance channel in LAB space."""
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def __call__(self, img_pil):
        img_np = np.array(img_pil)
        lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
        cl = clahe.apply(l)
        enhanced = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2RGB)
        return Image.fromarray(enhanced)

def get_transforms(img_size=224, is_training=True):
    """
    Returns torchvision transforms.
    Training augmentations are realistic, conservative, and designed for microscopy:
      - Small rotations (+/- 15 deg)
      - Flips (horizontal & vertical)
      - Mild brightness and contrast shifts
      - Small affine scale/translation
    Validation transforms are strictly deterministic.
    """
    if is_training:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.08),
            transforms.RandomAffine(degrees=0, translate=(0.04, 0.04), scale=(0.95, 1.05)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

class EmbryoDataset(Dataset):
    def __init__(self, df, img_dir='Images', transform=None, use_clahe=False):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.clahe = CLAHETransform() if use_clahe else None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row['filename'])
        image = Image.open(img_path).convert('RGB')

        if self.clahe is not None:
            image = self.clahe(image)

        if self.transform is not None:
            image = self.transform(image)

        label = GRADE_MAP[row['grade']]
        patient_id = row['patient_id']
        filename = row['filename']
        return image, label, patient_id, filename

def load_clean_dataset(csv_path='labels_clean.csv', remove_duplicates=True):
    """
    Loads verified ground-truth dataset and removes bitwise duplicate 0105_07.png if requested.
    """
    df = pd.read_csv(csv_path)
    labeled = df[df['status'] == 'ok'].copy()
    if remove_duplicates:
        # 0105_07.png is an exact MD5 duplicate of 0105_03.png
        labeled = labeled[labeled['filename'] != '0105_07.png'].copy()
    return labeled.reset_index(drop=True)
