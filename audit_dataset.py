import os
import sys
import hashlib
import pandas as pd
import numpy as np
from PIL import Image
import torch
import torchvision

sys.stdout.reconfigure(line_buffering=True)

print("=== PHASE 1 & 2: EMBRYO DATASET & SYSTEM AUDIT ===", flush=True)

# 1. Framework & Hardware
print("\n[1] Framework & Hardware Environment:", flush=True)
print(f"  PyTorch Version: {torch.__version__}", flush=True)
print(f"  Torchvision Version: {torchvision.__version__}", flush=True)
print(f"  CUDA Available: {torch.cuda.is_available()}", flush=True)
if torch.cuda.is_available():
    print(f"  Device Name: {torch.cuda.get_device_name(0)}", flush=True)
    vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"  Total VRAM: {vram:.2f} GB", flush=True)

# 2. Raw Image Files Audit
img_dir = "Images"
all_files = [f for f in os.listdir(img_dir) if os.path.isfile(os.path.join(img_dir, f))] if os.path.exists(img_dir) else []
print(f"\n[2] Raw Image Directory ('{img_dir}/'):", flush=True)
print(f"  Total Image Files Found: {len(all_files)}", flush=True)

# 3. Label File Audit
orig_csv = "Blastocyst labels - labels.csv"
df_orig = pd.read_csv(orig_csv)
print(f"\n[3] Master Label File ('{orig_csv}'):", flush=True)
print(f"  Total CSV Entries: {len(df_orig)}", flush=True)
print(f"  Unique Filenames in CSV: {df_orig['filename'].nunique()}", flush=True)
print("  Status breakdown in raw CSV:")
for k, v in df_orig['status'].value_counts(dropna=False).items():
    print(f"    - {k}: {v}")

print("  Grade breakdown in raw CSV (all entries):")
for k, v in df_orig['grade'].value_counts(dropna=False).items():
    print(f"    - {k}: {v}")

# 4. Verified Ground-Truth Labeled Images
df_labeled = df_orig[df_orig['status'] == 'ok'].copy()
print(f"\n[4] Ground-Truth Labeled Cohort (status == 'ok'):", flush=True)
print(f"  Total Verified Labeled Embryos: {len(df_labeled)}", flush=True)
print("  Class Distribution:")
for k, v in df_labeled['grade'].value_counts().items():
    print(f"    - {k}: {v} ({v/len(df_labeled)*100:.1f}%)")

# Check missing images
missing = [f for f in df_labeled['filename'] if not os.path.exists(os.path.join(img_dir, f))]
print(f"  Missing labeled images in Images/ folder: {len(missing)}", flush=True)

# 5. Image Dimensions, Modes, and Integrity Check for the 324 Labeled Images
sizes = set()
modes = set()
corrupted = []
hashes = {}
duplicates = []

for idx, row in df_labeled.iterrows():
    fpath = os.path.join(img_dir, row['filename'])
    try:
        with open(fpath, 'rb') as f:
            h = hashlib.md5(f.read()).hexdigest()
        if h in hashes:
            duplicates.append((row['filename'], hashes[h]))
        else:
            hashes[h] = row['filename']

        with Image.open(fpath) as im:
            sizes.add(im.size)
            modes.add(im.mode)
    except Exception as e:
        corrupted.append((row['filename'], str(e)))

print(f"\n[5] Image Geometry & Integrity (324 Labeled Images):", flush=True)
print(f"  Unique Dimensions (W x H): {sizes}", flush=True)
print(f"  Color Modes: {modes}", flush=True)
print(f"  Corrupted / Unreadable Images: {len(corrupted)}", flush=True)
print(f"  Exact Bitwise Duplicate Images (Identical MD5): {len(duplicates)}", flush=True)
if duplicates:
    for d in duplicates[:5]:
        print(f"    - {d[0]} is identical to {d[1]}")

# 6. Patient ID Grouping Analysis
df_labeled['patient_id'] = df_labeled['filename'].apply(lambda x: int(x.split('_')[0]))
n_patients = df_labeled['patient_id'].nunique()
print(f"\n[6] Patient Cohort & Multi-Frame Grouping:", flush=True)
print(f"  Total Unique Patients / Embryos: {n_patients}", flush=True)
counts = df_labeled.groupby('patient_id').size()
print(f"  Mean Frames per Patient: {counts.mean():.2f} (Min: {counts.min()}, Max: {counts.max()})", flush=True)
print("  Breakdown of Frames per Patient:")
for n_frames, count in counts.value_counts().sort_index().items():
    print(f"    - {count} patients have {n_frames} frame(s)")

# 7. Pretrained Checkpoints Available
print(f"\n[7] Local Pretrained PyTorch Checkpoints in Cache:", flush=True)
cache_dir = os.path.expanduser(r"~\.cache\torch\hub\checkpoints")
if os.path.exists(cache_dir):
    for f in os.listdir(cache_dir):
        fp = os.path.join(cache_dir, f)
        if os.path.isfile(fp):
            sz_mb = os.path.getsize(fp) / (1024**2)
            print(f"    - {f} ({sz_mb:.1f} MB)")
else:
    print("    - Cache directory not found")

print("\nAudit complete!", flush=True)
