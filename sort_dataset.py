import os
import shutil
import csv

def sort_dataset(
    csv_file="Blastocyst labels - labels.csv",
    src_dir="Images",
    dest_dir="sorted_dataset",
    include_unlabeled=True
):
    print("=" * 60)
    print("Sorting Embryo Dataset by Grade Labels")
    print("=" * 60)

    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"CSV file not found: {csv_file}")
    if not os.path.exists(src_dir):
        raise FileNotFoundError(f"Source image folder not found: {src_dir}")

    counts = {}
    with open(csv_file, mode="r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = row.get("filename", "").strip()
            grade = row.get("grade", "").strip()
            
            if not grade:
                if not include_unlabeled:
                    continue
                category = "Unlabeled"
            else:
                category = grade  # e.g., Grade-1, Grade-2, Grade-3

            target_folder = os.path.join(dest_dir, category)
            os.makedirs(target_folder, exist_ok=True)

            src_path = os.path.join(src_dir, filename)
            dest_path = os.path.join(target_folder, filename)

            if os.path.exists(src_path):
                # Copying so original images remain intact
                shutil.copy2(src_path, dest_path)
                counts[category] = counts.get(category, 0) + 1
            else:
                print(f"Warning: File not found: {src_path}")

    print("\nDataset sorting complete! Summary:")
    total = 0
    for cat in sorted(counts.keys()):
        print(f"  {cat}: {counts[cat]} images")
        total += counts[cat]
    print(f"Total processed images: {total}")
    print(f"Output folder: {os.path.abspath(dest_dir)}")
    print("=" * 60)

if __name__ == "__main__":
    sort_dataset()
