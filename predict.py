import sys
import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image, ImageOps, ImageEnhance
import numpy as np

try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False

class CLAHETransform:
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def __call__(self, img):
        if HAS_OPENCV:
            np_img = np.array(img)
            lab = cv2.cvtColor(np_img, cv2.COLOR_RGB2LAB)
            clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            return Image.fromarray(rgb)
        else:
            img = ImageOps.autocontrast(img, cutoff=1)
            enhancer = ImageEnhance.Contrast(img)
            return enhancer.enhance(1.15)

def load_efficientnet(weights_path="efficientnet_embryo_best.pth", num_classes=3, device="cuda"):
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, 256),
        nn.SiLU(),
        nn.Dropout(p=0.2),
        nn.Linear(256, num_classes)
    )
    if os.path.exists(weights_path):
        state_dict = torch.load(weights_path, map_location=device, weights_only=False)
        model.load_state_dict(state_dict)
        print(f"Loaded EfficientNet-B0 weights from {weights_path}")
    else:
        raise FileNotFoundError(f"Model weights not found: {weights_path}")
    model = model.to(device)
    model.eval()
    return model

def predict_image(image, model, device="cuda", img_size=288, use_tta=True):
    classes = ["Grade-1", "Grade-2", "Grade-3"]
    base_t = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        CLAHETransform(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    img = image.convert('RGB')
    
    with torch.no_grad():
        if use_tta:
            views = [
                img,
                img.transpose(Image.FLIP_LEFT_RIGHT),
                img.transpose(Image.FLIP_TOP_BOTTOM),
                img.transpose(Image.ROTATE_90)
            ]
            batch_views = torch.stack([base_t(v) for v in views]).to(device)
            outs = model(batch_views)
            probs = torch.softmax(outs, dim=1).mean(dim=0).cpu().numpy()
        else:
            input_tensor = base_t(img).unsqueeze(0).to(device)
            outs = model(input_tensor)
            probs = torch.softmax(outs, dim=1).cpu().numpy()[0]

        pred_idx = int(probs.argmax())
        
    return classes[pred_idx], probs

def predict_embryo(image_path, model, device="cuda", img_size=288, use_tta=True):
    img = Image.open(image_path).convert('RGB')
    predicted_grade, probs = predict_image(img, model, device=device, img_size=img_size, use_tta=use_tta)
    classes = ["Grade-1", "Grade-2", "Grade-3"]

    print(f"\n========================================================")
    print(f"Prediction for: {image_path}")
    print(f"Predicted Grade:  {predicted_grade} (Confidence: {probs[classes.index(predicted_grade)]*100:.2f}%)")
    print(f"========================================================")
    print("Class Probabilities:")
    for c, p in zip(classes, probs):
        bar = "#" * int(p * 25)
        print(f"  {c:7s}: {p*100:6.2f}%  [{bar:<25}]")
    print(f"========================================================\n")
        
    return predicted_grade, probs

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
    else:
        sample_dir = os.path.join("balanced_expanded_dataset", "Grade-1")
        if os.path.exists(sample_dir) and os.listdir(sample_dir):
            img_path = os.path.join(sample_dir, os.listdir(sample_dir)[0])
        else:
            img_path = "Images/0004_01.png"
            
    model = load_efficientnet(device=device)
    predict_embryo(img_path, model, device=device, use_tta=True)
