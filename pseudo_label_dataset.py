import os
import shutil
import csv
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

def load_vgg16(weights_path="vgg16_embryo_best.pth", num_classes=3, device="cuda"):
    model = models.vgg16(weights=None)
    in_features = model.classifier[6].in_features
    model.classifier[6] = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(in_features, num_classes)
    )
    if os.path.exists(weights_path):
        state_dict = torch.load(weights_path, map_location=device, weights_only=False)
        model.load_state_dict(state_dict)
        print(f"Successfully loaded model weights from {weights_path}")
    else:
        raise FileNotFoundError(f"Model weights not found: {weights_path}")
    model = model.to(device)
    model.eval()
    return model

def pseudo_label_unlabeled(
    src_unlabeled_dir="sorted_dataset/Unlabeled",
    original_sorted_dir="sorted_dataset",
    output_dir="expanded_dataset",
    confidence_threshold=0.85,
    device="cuda"
):
    classes = ["Grade-1", "Grade-2", "Grade-3"]
    model = load_vgg16(device=device)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    print("=" * 65)
    print("STEP 1: Copying Original Ground-Truth Labeled Images")
    print("=" * 65)
    orig_counts = {}
    for c in classes:
        src_cls = os.path.join(original_sorted_dir, c)
        dst_cls = os.path.join(output_dir, c)
        os.makedirs(dst_cls, exist_ok=True)
        if os.path.exists(src_cls):
            files = [f for f in os.listdir(src_cls) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            for f in files:
                shutil.copy2(os.path.join(src_cls, f), os.path.join(dst_cls, f))
            orig_counts[c] = len(files)
            print(f"  {c}: {len(files)} ground-truth images copied.")

    print("\n" + "=" * 65)
    print(f"STEP 2: Classifying Unlabeled Images (Confidence Threshold > {confidence_threshold*100:.0f}%)")
    print("=" * 65)

    if not os.path.exists(src_unlabeled_dir):
        raise FileNotFoundError(f"Unlabeled directory not found: {src_unlabeled_dir}")

    unlabeled_files = [f for f in os.listdir(src_unlabeled_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    print(f"Found {len(unlabeled_files)} unlabeled images in {src_unlabeled_dir}")

    pseudo_counts = {c: 0 for c in classes}
    low_confidence_count = 0
    records = []

    os.makedirs(os.path.join(output_dir, "Remaining_Unlabeled"), exist_ok=True)

    with torch.no_grad():
        for idx, filename in enumerate(unlabeled_files):
            img_path = os.path.join(src_unlabeled_dir, filename)
            try:
                img = Image.open(img_path).convert('RGB')
                tensor = transform(img).unsqueeze(0).to(device)
                outputs = model(tensor)
                probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
                pred_idx = probs.argmax()
                max_conf = float(probs[pred_idx])
                pred_class = classes[pred_idx]

                record = {
                    "filename": filename,
                    "predicted_grade": pred_class,
                    "confidence": f"{max_conf:.4f}",
                    "prob_grade_1": f"{probs[0]:.4f}",
                    "prob_grade_2": f"{probs[1]:.4f}",
                    "prob_grade_3": f"{probs[2]:.4f}",
                    "accepted": False
                }

                if max_conf >= confidence_threshold:
                    record["accepted"] = True
                    pseudo_counts[pred_class] += 1
                    dest_path = os.path.join(output_dir, pred_class, f"pseudo_{filename}")
                    shutil.copy2(img_path, dest_path)
                else:
                    low_confidence_count += 1
                    dest_path = os.path.join(output_dir, "Remaining_Unlabeled", filename)
                    shutil.copy2(img_path, dest_path)

                records.append(record)

                if (idx + 1) % 400 == 0 or (idx + 1) == len(unlabeled_files):
                    print(f"  Processed {idx + 1}/{len(unlabeled_files)} images...")

            except Exception as e:
                print(f"Error processing {filename}: {e}")

    # Save summary CSV
    csv_out = "pseudo_labels_summary.csv"
    with open(csv_out, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["filename", "predicted_grade", "confidence", "prob_grade_1", "prob_grade_2", "prob_grade_3", "accepted"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    print("\n" + "=" * 65)
    print("Pseudo-Labeling Summary:")
    print("=" * 65)
    total_accepted = sum(pseudo_counts.values())
    print(f"Total Unlabeled Images Processed: {len(unlabeled_files)}")
    print(f"Accepted High-Confidence Images (>= {confidence_threshold*100:.0f}%): {total_accepted}")
    print(f"Remaining Low-Confidence Images (< {confidence_threshold*100:.0f}%):  {low_confidence_count}")
    print("\nPseudo-labeled additions by class:")
    for c in classes:
        print(f"  {c}: +{pseudo_counts[c]} pseudo-labeled images")

    print("\nTotal Expanded Dataset (Ground-Truth + High-Confidence Pseudo):")
    total_dataset = 0
    for c in classes:
        count = len(os.listdir(os.path.join(output_dir, c)))
        print(f"  {c}: {count} total images ({orig_counts.get(c, 0)} original + {pseudo_counts[c]} pseudo)")
        total_dataset += count
    print(f"Total labeled images in {output_dir}: {total_dataset}")
    print(f"Detailed predictions saved to: {csv_out}")
    print("=" * 65)

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on device: {device}")
    pseudo_label_unlabeled(
        src_unlabeled_dir="sorted_dataset/Unlabeled",
        original_sorted_dir="sorted_dataset",
        output_dir="expanded_dataset",
        confidence_threshold=0.85,
        device=device
    )
