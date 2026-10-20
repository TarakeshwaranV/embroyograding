import os
import copy
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from torchvision.models import VGG16_Weights
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=20),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return train_transform, val_transform

class TransformedDataset(torch.utils.data.Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, idx):
        x, y = self.subset[idx]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)

def prepare_dataloaders(data_dir="balanced_dataset", batch_size=16, test_size=0.2, random_state=42):
    target_classes = ["Grade-1", "Grade-2", "Grade-3"]
    
    base_dataset = datasets.ImageFolder(root=data_dir)
    class_to_idx = base_dataset.class_to_idx
    print("=" * 65)
    print("Dataset Loading & Splitting")
    print("=" * 65)
    print("Detected folder classes:", class_to_idx)
    
    # Filter indices for Grade-1, Grade-2, Grade-3 only
    valid_class_indices = {class_to_idx[c] for c in target_classes if c in class_to_idx}
    selected_indices = [i for i, (_, label) in enumerate(base_dataset.samples) if label in valid_class_indices]
    
    # Map the labels to 0, 1, 2
    label_mapping = {class_to_idx[c]: i for i, c in enumerate(target_classes)}
    
    # Filter samples
    filtered_samples = [(base_dataset.samples[i][0], label_mapping[base_dataset.samples[i][1]]) for i in selected_indices]
    base_dataset.samples = filtered_samples
    base_dataset.targets = [s[1] for s in filtered_samples]
    base_dataset.classes = target_classes
    base_dataset.class_to_idx = {c: i for i, c in enumerate(target_classes)}
    
    print(f"Total balanced embryo samples: {len(base_dataset)}")
    for c, i in base_dataset.class_to_idx.items():
        print(f"  Class {i} -> {c}: {sum(1 for t in base_dataset.targets if t == i)} images")
        
    targets = base_dataset.targets
    indices = list(range(len(base_dataset)))
    
    train_idx, val_idx = train_test_split(
        indices,
        test_size=test_size,
        stratify=targets,
        random_state=random_state
    )
    
    train_transform, val_transform = get_transforms()
    
    raw_train_subset = Subset(base_dataset, train_idx)
    raw_val_subset = Subset(base_dataset, val_idx)
    
    train_dataset = TransformedDataset(raw_train_subset, transform=train_transform)
    val_dataset = TransformedDataset(raw_val_subset, transform=val_transform)
    
    # num_workers=0 ensures 100% compatibility on Windows without subprocess spawning overhead
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    
    print(f"Training set: {len(train_dataset)} images | Validation set: {len(val_dataset)} images")
    return train_loader, val_loader, target_classes

def build_vgg16(num_classes=3, device="cuda"):
    print("\n" + "=" * 65)
    print("VGG16 Model Architecture & Activation Functions")
    print("=" * 65)
    print("Backbone: Pretrained VGG16 (ImageNet weights)")
    print("Convolutional Activation Function: ReLU (Rectified Linear Unit)")
    print("Classifier Head Activation Function: ReLU between dense layers + Softmax for 3-class output")
    
    model = models.vgg16(weights=VGG16_Weights.DEFAULT)
    
    # Freeze feature extraction layers initially
    for param in model.features.parameters():
        param.requires_grad = False
        
    # Replace final classification head
    in_features = model.classifier[6].in_features
    model.classifier[6] = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(in_features, num_classes)
    )
    
    model = model.to(device)
    print("Model successfully loaded onto GPU / device:", device)
    return model

def train_model(model, train_loader, val_loader, class_names, epochs=20, lr=1e-4, device="cuda"):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    scaler = torch.cuda.amp.GradScaler()
    
    best_val_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    print("\n" + "=" * 65)
    print("Starting Model Training on CUDA")
    print("=" * 65)
    print("Phase 1: Training Classifier Head (Epochs 1-10)...")
    start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        if epoch == 11:
            print("\nPhase 2: Fine-Tuning Top Convolutional Block (VGG16 features[24:])...")
            for param in model.features[24:].parameters():
                param.requires_grad = True
            optimizer = optim.AdamW([
                {'params': model.features[24:].parameters(), 'lr': lr * 0.1},
                {'params': model.classifier.parameters(), 'lr': lr}
            ], weight_decay=1e-4)
            
        # Training phase
        model.train()
        running_loss = 0.0
        running_corrects = 0
        total_train = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data).item()
            total_train += inputs.size(0)
            
        epoch_train_loss = running_loss / total_train
        epoch_train_acc = running_corrects / total_train
        
        # Validation phase
        model.eval()
        running_val_loss = 0.0
        running_val_corrects = 0
        total_val = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    
                _, preds = torch.max(outputs, 1)
                running_val_loss += loss.item() * inputs.size(0)
                running_val_corrects += torch.sum(preds == labels.data).item()
                total_val += inputs.size(0)
                
        epoch_val_loss = running_val_loss / total_val
        epoch_val_acc = running_val_corrects / total_val
        
        scheduler.step(epoch_val_loss)
        
        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)
        
        print(f"Epoch [{epoch:02d}/{epochs:02d}] "
              f"Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc * 100:.2f}% | "
              f"Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc * 100:.2f}%")
        
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(best_model_wts, "vgg16_embryo_best.pth")
            
    time_elapsed = time.time() - start_time
    print(f"\nTraining completed in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Best Validation Accuracy: {best_val_acc * 100:.2f}%")
    print("Saved best model weights to: vgg16_embryo_best.pth")
    
    model.load_state_dict(best_model_wts)
    return model, history

def evaluate_model(model, val_loader, class_names, device="cuda"):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            
    print("\n" + "=" * 65)
    print("Validation Classification Report:")
    print("=" * 65)
    print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))
    
    cm = confusion_matrix(all_labels, all_preds)
    print("Confusion Matrix:")
    print(cm)
    
    # Save Confusion Matrix figure
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Embryo Grade Classification Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
                     
    plt.tight_layout()
    plt.ylabel('True Grade')
    plt.xlabel('Predicted Grade')
    plt.savefig('confusion_matrix.png', dpi=300)
    plt.close()
    print("Saved confusion matrix to confusion_matrix.png")

def plot_curves(history):
    epochs = range(1, len(history['train_loss']) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Loss
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    ax1.plot(epochs, history['val_loss'], 'r--', label='Val Loss')
    ax1.set_title('Training & Validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy
    ax2.plot(epochs, [a * 100 for a in history['train_acc']], 'b-', label='Train Acc')
    ax2.plot(epochs, [a * 100 for a in history['val_acc']], 'r--', label='Val Acc')
    ax2.set_title('Training & Validation Accuracy (%)')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=300)
    plt.close()
    print("Saved training curves to training_curves.png")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"Device Name: {torch.cuda.get_device_name(0)}")
        print(f"Available VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
        
    train_loader, val_loader, class_names = prepare_dataloaders(
        data_dir="balanced_dataset",
        batch_size=16,
        test_size=0.2,
        random_state=42
    )
    
    model = build_vgg16(num_classes=len(class_names), device=device)
    model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        class_names=class_names,
        epochs=20,
        lr=1e-4,
        device=device
    )
    
    evaluate_model(model, val_loader, class_names, device=device)
    plot_curves(history)
    print("\nAll training and evaluation steps successfully completed!")

if __name__ == "__main__":
    main()
