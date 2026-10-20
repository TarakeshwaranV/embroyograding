import os
import shutil
import random
from PIL import Image, ImageEnhance

def augment_image(img):
    """Applies random microscopy-safe biological transformations."""
    aug_choice = random.choice(['flip_lr', 'flip_tb', 'rot90', 'rot180', 'rot270', 'rotate_arb'])
    if aug_choice == 'flip_lr':
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    elif aug_choice == 'flip_tb':
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
    elif aug_choice == 'rot90':
        img = img.transpose(Image.ROTATE_90)
    elif aug_choice == 'rot180':
        img = img.transpose(Image.ROTATE_180)
    elif aug_choice == 'rot270':
        img = img.transpose(Image.ROTATE_270)
    elif aug_choice == 'rotate_arb':
        angle = random.uniform(-25, 25)
        img = img.rotate(angle, resample=Image.BILINEAR)

    # Lighting / contrast adjustment
    if random.random() > 0.4:
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(random.uniform(0.88, 1.12))
    if random.random() > 0.4:
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(random.uniform(0.88, 1.15))
    if random.random() > 0.5:
        enhancer = ImageEnhance.Sharpness(img)
        img = enhancer.enhance(random.uniform(0.9, 1.2))

    return img

def create_balanced_expanded(
    src_dir="expanded_dataset",
    dest_dir="balanced_expanded_dataset",
    target_count_per_class=250
):
    classes = ["Grade-1", "Grade-2", "Grade-3"]
    random.seed(42)

    print("=" * 65)
    print(f"Balancing Expanded Dataset to {target_count_per_class} images per class")
    print("=" * 65)

    counts = {}
    for cls in classes:
        src_cls_dir = os.path.join(src_dir, cls)
        dst_cls_dir = os.path.join(dest_dir, cls)
        os.makedirs(dst_cls_dir, exist_ok=True)

        existing_files = [f for f in os.listdir(src_cls_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        base_count = len(existing_files)
        print(f"\nProcessing '{cls}': {base_count} existing images (ground-truth + pseudo).")

        # Copy existing images
        for f in existing_files:
            shutil.copy2(os.path.join(src_cls_dir, f), os.path.join(dst_cls_dir, f))

        # Generate augmented images to reach target_count_per_class
        needed = target_count_per_class - base_count
        if needed > 0:
            print(f"  Generating {needed} augmented images to reach {target_count_per_class}...")
            aug_idx = 0
            while aug_idx < needed:
                sample_file = random.choice(existing_files)
                try:
                    with Image.open(os.path.join(src_cls_dir, sample_file)) as img:
                        img = img.convert('RGB')
                        aug_img = augment_image(img)
                        base_name, ext = os.path.splitext(sample_file)
                        out_name = f"aug_{aug_idx:03d}_{base_name}{ext}"
                        aug_img.save(os.path.join(dst_cls_dir, out_name))
                        aug_idx += 1
                except Exception as e:
                    print(f"Error augmenting {sample_file}: {e}")
        else:
            print(f"  Already has {base_count} images (>= {target_count_per_class}).")

        final_count = len(os.listdir(dst_cls_dir))
        counts[cls] = final_count
        print(f"  Final count for '{cls}': {final_count} images.")

    print("\n" + "=" * 65)
    print("Dataset balancing complete!")
    print(f"Balanced expanded dataset saved to: {os.path.abspath(dest_dir)}")
    total = sum(counts.values())
    for cls in classes:
        print(f"  {cls}: {counts[cls]} images")
    print(f"Total dataset size: {total} images (perfectly balanced 1:1:1)")
    print("=" * 65)

if __name__ == "__main__":
    create_balanced_expanded(
        src_dir="expanded_dataset",
        dest_dir="balanced_expanded_dataset",
        target_count_per_class=250
    )
