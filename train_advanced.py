import os
import copy
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import datasets, transforms, models
from torchvision.models import EfficientNet_B0_Weights
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, cohen_kappa_score
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps, ImageEnhance

try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False

class CLAHETransform:
    """Applies Contrast Limited Adaptive Histogram Equalization for microscopy images."""
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

class OrdinalSoftCrossEntropyLoss(nn.Module):
    """
    Ordinal soft cross-entropy for embryo staging (Grade-1 < Grade-2 < Grade-3).
    Penalizes extreme errors (Grade-1 -> Grade-3) more heavily than adjacent grades.
    """
    def __init__(self, num_classes=3, smoothing=0.15):
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, logits, targets):
        log_probs = self.log_softmax(logits)
        batch_size = targets.size(0)
        target_dist = torch.zeros_like(logits)
        for i in range(batch_size):
            y = targets[i].item()
            if y == 0:
                target_dist[i, 0] = 1.0 - self.smoothing
                target_dist[i, 1] = self.smoothing
            elif y == 1:
                target_dist[i, 0] = self.smoothing / 2.0
                target_dist[i, 1] = 1.0 - self.smoothing
                target_dist[i, 2] = self.smoothing / 2.0
            elif y == 2:
                target_dist[i, 1] = self.smoothing
                target_dist[i, 2] = 1.0 - self.smoothing

        loss = -(target_dist * log_probs).sum(dim=1).mean()
        return loss

def get_transforms(img_size=288):
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        CLAHETransform(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=25),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        CLAHETransform(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return train_transform, val_transform

class TransformedDataset(Dataset):
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

def build_efficientnet(num_classes=3, device="cuda"):
    print("Loading Pretrained EfficientNet-B0 (ImageNet weights)...")
    model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, 256),
        nn.SiLU(),
        nn.Dropout(p=0.2),
        nn.Linear(256, num_classes)
    )
    return model.to(device)

def train_efficientnet(model, train_loader, val_loader, class_names, epochs=20, lr=3e-4, device="cuda"):
    criterion = OrdinalSoftCrossEntropyLoss(num_classes=len(class_names), smoothing=0.15)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda') if device == "cuda" else None

    best_val_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    print("\n" + "=" * 65)
    print("Training EfficientNet-B0 on Expanded & Balanced Embryo Dataset")
    print("=" * 65)
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        running_corrects = 0
        total_train = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            if device == "cuda":
                with torch.amp.autocast('cuda'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

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
                if device == "cuda":
                    with torch.amp.autocast('cuda'):
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                else:
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                _, preds = torch.max(outputs, 1)
                running_val_loss += loss.item() * inputs.size(0)
                running_val_corrects += torch.sum(preds == labels.data).item()
                total_val += inputs.size(0)

        epoch_val_loss = running_val_loss / total_val
        epoch_val_acc = running_val_corrects / total_val

        scheduler.step()

        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)

        print(f"Epoch [{epoch:02d}/{epochs:02d}] "
              f"Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc*100:.2f}% | "
              f"Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc*100:.2f}% (LR: {scheduler.get_last_lr()[0]:.2e})")

        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(best_model_wts, "efficientnet_embryo_best.pth")

    time_elapsed = time.time() - start_time
    print(f"\nTraining completed in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Best Validation Accuracy: {best_val_acc * 100:.2f}%")
    print("Saved best model weights to: efficientnet_embryo_best.pth")

    model.load_state_dict(best_model_wts)
    return model, history

def evaluate_with_tta(model, val_dataset, class_names, img_size=288, device="cuda"):
    model.eval()
    all_preds_std = []
    all_preds_tta = []
    all_labels = []
    all_probs_tta = []

    print("\n" + "=" * 65)
    print("Evaluating Model with Test-Time Augmentation (TTA)")
    print("=" * 65)

    base_t = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        CLAHETransform(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    with torch.no_grad():
        for i in range(len(val_dataset)):
            img, label = val_dataset.subset[i]
            all_labels.append(label)

            views = [
                img,
                img.transpose(Image.FLIP_LEFT_RIGHT),
                img.transpose(Image.FLIP_TOP_BOTTOM),
                img.transpose(Image.ROTATE_90)
            ]
            batch_views = torch.stack([base_t(v) for v in views]).to(device)

            if device == "cuda":
                with torch.amp.autocast('cuda'):
                    outs = model(batch_views)
            else:
                outs = model(batch_views)

            probs = torch.softmax(outs, dim=1).mean(dim=0).cpu().numpy()
            pred_tta = np.argmax(probs)
            pred_std = torch.softmax(outs[0:1], dim=1).cpu().numpy()[0].argmax()

            all_preds_std.append(pred_std)
            all_preds_tta.append(pred_tta)
            all_probs_tta.append(probs)

    all_labels = np.array(all_labels)
    all_preds_std = np.array(all_preds_std)
    all_preds_tta = np.array(all_preds_tta)
    all_probs_tta = np.array(all_probs_tta)

    acc_std = accuracy_score(all_labels, all_preds_std)
    acc_tta = accuracy_score(all_labels, all_preds_tta)
    auc_ovr = roc_auc_score(all_labels, all_probs_tta, multi_class='ovr')
    kappa = cohen_kappa_score(all_labels, all_preds_tta)

    print(f"Standard Single-View Accuracy: {acc_std * 100:.2f}%")
    print(f"TTA Enhanced Accuracy:        {acc_tta * 100:.2f}%")
    print(f"Multiclass ROC-AUC (OvR):     {auc_ovr * 100:.2f}%")
    print(f"Cohen's Kappa Score:          {kappa:.4f}")

    print("\nClassification Report (with TTA):")
    print(classification_report(all_labels, all_preds_tta, target_names=class_names, digits=4))

    cm = confusion_matrix(all_labels, all_preds_tta)
    print("Confusion Matrix:")
    print(cm)

    # Plot Confusion Matrix
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Greens)
    plt.title('EfficientNet-B0 (TTA) Confusion Matrix')
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
    plt.savefig('confusion_matrix_efficientnet.png', dpi=300)
    plt.close()
    print("Saved confusion matrix plot to: confusion_matrix_efficientnet.png")

    return acc_tta, all_probs_tta

def plot_training_curves(history):
    epochs = range(1, len(history['train_loss']) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    ax1.plot(epochs, history['val_loss'], 'r--', label='Val Loss')
    ax1.set_title('EfficientNet-B0 Loss (Ordinal Loss)')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(epochs, [a * 100 for a in history['train_acc']], 'b-', label='Train Acc')
    ax2.plot(epochs, [a * 100 for a in history['val_acc']], 'r--', label='Val Acc')
    ax2.set_title('EfficientNet-B0 Accuracy (%)')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig('training_curves_efficientnet.png', dpi=300)
    plt.close()
    print("Saved training curves to: training_curves_efficientnet.png")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using compute device: {device}")
    if torch.cuda.is_available():
        print(f"Device Name: {torch.cuda.get_device_name(0)}")
        print(f"Available VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")

    target_classes = ["Grade-1", "Grade-2", "Grade-3"]
    data_dir = "balanced_expanded_dataset" if os.path.exists("balanced_expanded_dataset") else "expanded_dataset"

    base_dataset = datasets.ImageFolder(root=data_dir)
    c2i = base_dataset.class_to_idx

    valid_class_indices = {c2i[c] for c in target_classes if c in c2i}
    selected_indices = [i for i, (_, label) in enumerate(base_dataset.samples) if label in valid_class_indices]
    label_mapping = {c2i[c]: i for i, c in enumerate(target_classes)}

    filtered_samples = [(base_dataset.samples[i][0], label_mapping[base_dataset.samples[i][1]]) for i in selected_indices]
    base_dataset.samples = filtered_samples
    base_dataset.targets = [s[1] for s in filtered_samples]
    base_dataset.classes = target_classes
    base_dataset.class_to_idx = {c: i for i, c in enumerate(target_classes)}

    targets = base_dataset.targets
    indices = list(range(len(base_dataset)))

    train_idx, val_idx = train_test_split(indices, test_size=0.2, stratify=targets, random_state=42)

    img_size = 288
    train_transform, val_transform = get_transforms(img_size=img_size)

    raw_train_subset = Subset(base_dataset, train_idx)
    raw_val_subset = Subset(base_dataset, val_idx)

    train_dataset = TransformedDataset(raw_train_subset, transform=train_transform)
    val_dataset = TransformedDataset(raw_val_subset, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0, pin_memory=True)

    print(f"Dataset Loaded: {len(train_dataset)} Train | {len(val_dataset)} Val (Resolution: {img_size}x{img_size})")

    model = build_efficientnet(num_classes=3, device=device)
    model, history = train_efficientnet(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        class_names=target_classes,
        epochs=20,
        lr=3e-4,
        device=device
    )

    evaluate_with_tta(model, val_dataset, target_classes, img_size=img_size, device=device)
    plot_training_curves(history)
    print("\nAll training and evaluation steps successfully completed!")

if __name__ == "__main__":
    main()
