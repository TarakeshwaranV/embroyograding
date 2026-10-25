"""
Multi-Focal Embryo Bag Dataset for Multiple Instance Learning (MIL)
==================================================================
Groups all focal slice images belonging to each embryo / patient into a "Bag".
During training, slices receive realistic, independent on-the-fly augmentations.
During validation, slices are evaluated deterministically.
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

def get_mil_transforms(img_size=224, is_training=True):
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

class EmbryoBagDataset(Dataset):
    """
    Each item represents a single patient's embryo containing K focal slices.
    """
    def __init__(self, df, img_dir='Images', transform=None, use_clahe=True):
        self.img_dir = img_dir
        self.transform = transform
        self.clahe = CLAHETransform() if use_clahe else None

        # Group by patient_id
        self.bags = []
        for patient_id, group in df.groupby('patient_id'):
            self.bags.append({
                'patient_id': patient_id,
                'grade': group['grade'].iloc[0],
                'label': GRADE_MAP[group['grade'].iloc[0]],
                'filenames': list(group['filename'].values),
                'fold': group['fold'].iloc[0]
            })

    def __len__(self):
        return len(self.bags)

    def __getitem__(self, idx):
        bag = self.bags[idx]
        images_tensor = []

        for fname in bag['filenames']:
            img_path = os.path.join(self.img_dir, fname)
            img = Image.open(img_path).convert('RGB')
            if self.clahe is not None:
                img = self.clahe(img)
            if self.transform is not None:
                img_t = self.transform(img)
            else:
                img_t = transforms.ToTensor()(img)
            images_tensor.append(img_t)

        # Stack into shape (K, C, H, W) where K is number of focal slices
        images_tensor = torch.stack(images_tensor, dim=0)
        label = bag['label']
        patient_id = bag['patient_id']
        filenames = bag['filenames']

        return images_tensor, label, patient_id, filenames

def load_mil_dataframe(csv_path='labels_clean.csv'):
    df = pd.read_csv(csv_path)
    labeled = df[(df['status'] == 'ok') & (df['filename'] != '0105_07.png')].copy()
    return labeled.reset_index(drop=True)
