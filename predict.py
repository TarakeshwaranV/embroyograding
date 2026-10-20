import sys
import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from torchvision.models import VGG16_Weights
from PIL import Image

def load_model(weights_path="vgg16_embryo_best.pth", num_classes=3, device="cuda"):
    model = models.vgg16(weights=None)
    in_features = model.classifier[6].in_features
    model.classifier[6] = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(in_features, num_classes)
    )
    if os.path.exists(weights_path):
        state_dict = torch.load(weights_path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"Loaded weights from {weights_path}")
    else:
        print(f"Warning: Weights file {weights_path} not found.")
    model = model.to(device)
    model.eval()
    return model

def predict_embryo(image_path, model, device="cuda"):
    classes = ["Grade-1", "Grade-2", "Grade-3"]
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img = Image.open(image_path).convert('RGB')
    input_tensor = transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
        pred_idx = probs.argmax()
        
    print(f"\nPrediction for: {image_path}")
    print(f"Predicted Grade: {classes[pred_idx]} (Confidence: {probs[pred_idx]*100:.2f}%)")
    print("Probabilities:")
    for c, p in zip(classes, probs):
        print(f"  {c}: {p*100:.2f}%")
        
    return classes[pred_idx], probs

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
    else:
        # Default test on a sample image from Grade-1
        sample_dir = os.path.join("sorted_dataset", "Grade-1")
        if os.path.exists(sample_dir) and os.listdir(sample_dir):
            img_path = os.path.join(sample_dir, os.listdir(sample_dir)[0])
        else:
            img_path = "Images/0004_01.png"
            
    model = load_model(device=device)
    predict_embryo(img_path, model, device=device)
