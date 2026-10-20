import os
import shutil
import random
from PIL import Image, ImageEnhance, ImageFilter

def augment_image(img):
    """Applies a random combination of microscopy-safe augmentations."""
    # 1. Geometric transforms
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
        angle = random.uniform(-30, 30)
        img = img.rotate(angle, resample=Image.BILINEAR)

    # 2. Lighting / contrast adjustments (safe for biological imaging)
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

def create_balanced_dataset(
    src_dir="sorted_dataset",
    dest_dir="balanced_dataset",
    target_count_per_class=150
):
    print("=" * 65)
    print(f"Creating Balanced Dataset with Target: {target_count_per_class} images per class")
    print("=" * 65)

    classes = ["Grade-1", "Grade-2", "Grade-3"]
    random.seed(42)

    for cls in classes:
        src_cls_dir = os.path.join(src_dir, cls)
        dest_cls_dir = os.path.join(dest_dir, cls)
        os.makedirs(dest_cls_dir, exist_ok=True)

        if not os.path.exists(src_cls_dir):
            print(f"Warning: {src_cls_dir} does not exist!")
            continue

        original_files = [f for f in os.listdir(src_cls_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        orig_count = len(original_files)
        print(f"\nProcessing '{cls}': {orig_count} original images found.")

        # Copy original images
        for f in original_files:
            src_f = os.path.join(src_cls_dir, f)
            dest_f = os.path.join(dest_cls_dir, f)
            shutil.copy2(src_f, dest_f)

        # Augment to reach target_count_per_class
        needed = target_count_per_class - orig_count
        if needed > 0:
            print(f"  Generating {needed} augmented images to reach {target_count_per_class}...")
            aug_idx = 0
            while aug_idx < needed:
                sample_file = random.choice(original_files)
                src_img_path = os.path.join(src_cls_dir, sample_file)
                try:
                    with Image.open(src_img_path) as img:
                        img = img.convert('RGB')
                        aug_img = augment_image(img)
                        base_name, ext = os.path.splitext(sample_file)
                        out_name = f"aug_{aug_idx:03d}_{base_name}{ext}"
                        out_path = os.path.join(dest_cls_dir, out_name)
                        aug_img.save(out_path)
                        aug_idx += 1
                except Exception as e:
                    print(f"Error augmenting {sample_file}: {e}")
        else:
            print(f"  Already has {orig_count} images (>= {target_count_per_class}). No augmentation needed.")

        final_count = len(os.listdir(dest_cls_dir))
        print(f"  Final count for '{cls}': {final_count} images.")

    print("\n" + "=" * 65)
    print("Dataset balancing complete!")
    print(f"Balanced dataset saved to: {os.path.abspath(dest_dir)}")
    for cls in classes:
        p = os.path.join(dest_dir, cls)
        print(f"  {cls}: {len(os.listdir(p))} images")
    print("=" * 65)

if __name__ == "__main__":
    create_balanced_dataset()
