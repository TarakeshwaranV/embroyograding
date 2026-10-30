"""Blastocyst grading: one training script for all models, so every model is compared fairly.

Models:  efficientnet_b4 | resnet50 | convnext_tiny   (timm, ImageNet-pretrained)
Data:    labels_clinician_324.csv (324 labelled images, 5 patient-grouped folds)

Example:
  python train.py --model resnet50 --images "D:/Embryo Grading/images" --csv labels_clinician_324.csv
  python train.py --model convnext_tiny --images ... --folds 0 1     # only some folds
  python train.py --check-data --images ...                          # only verifies files, then exits
  python train.py --preview-aug --images ...                         # saves an augmentation figure

IMPORTANT: Augmentation is applied on the fly to TRAINING images only. Validation and test
images are never augmented, so reported numbers are on real, untouched images.
Everyone must keep --img-size, --seed, --epochs identical across models.
"""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import (accuracy_score, cohen_kappa_score, confusion_matrix,
                             f1_score, precision_recall_fscore_support, roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold

GRADES = ["Grade-1", "Grade-2", "Grade-3"]
G2I = {g: i for i, g in enumerate(GRADES)}
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def seed_everything(seed):
    import torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def index_images(root):
    """Map filename -> full path, searching all subfolders."""
    idx = {}
    for p in Path(root).rglob("*"):
        if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"):
            idx[p.name] = p
    return idx


def center_square(img):
    w, h = img.size
    s = min(w, h)
    l, t = (w - s) // 2, (h - s) // 2
    return img.crop((l, t, l + s, t + s))


def build_transforms(size):
    from torchvision import transforms as T
    train = T.Compose([
        T.Lambda(center_square),
        T.Resize((size, size)),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        # an embryo has no fixed orientation, so any rotation is valid; corners filled with neutral grey
        T.RandomAffine(degrees=180, translate=(0.05, 0.05), scale=(0.9, 1.1), fill=(128, 128, 128)),
        T.ColorJitter(brightness=0.1, contrast=0.1),
        T.RandomApply([T.GaussianBlur(5, sigma=(0.1, 1.5))], p=0.2),
        T.ToTensor(),
        T.Normalize(MEAN, STD),
    ])
    evalt = T.Compose([T.Lambda(center_square), T.Resize((size, size)), T.ToTensor(), T.Normalize(MEAN, STD)])
    return train, evalt


class DS:
    def __init__(self, df, paths, tfm):
        self.df, self.paths, self.tfm = df.reset_index(drop=True), paths, tfm

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = Image.open(self.paths[r["filename"]]).convert("RGB")
        return self.tfm(img), G2I[r["grade"]]


def make_model(name, pretrained):
    import timm
    return timm.create_model(name, pretrained=pretrained, num_classes=3, drop_rate=0.3)


def run_epoch(model, loader, device, loss_fn, opt=None, scaler=None):
    import torch
    train = opt is not None
    model.train(train)
    losses, ys, ps, probs = [], [], [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.set_grad_enabled(train):
            with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                out = model(x)
                loss = loss_fn(out, y)
        if train:
            opt.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            else:
                loss.backward(); opt.step()
        losses.append(loss.item() * len(y))
        pr = torch.softmax(out.float(), 1).detach().cpu().numpy()
        probs.append(pr); ps += pr.argmax(1).tolist(); ys += y.cpu().tolist()
    n = len(ys)
    return sum(losses) / n, np.array(ys), np.array(ps), np.concatenate(probs)


def metrics(y, p, probs):
    prec, rec, f1, _ = precision_recall_fscore_support(y, p, labels=[0, 1, 2], zero_division=0)
    try:
        auc = roc_auc_score(y, probs, multi_class="ovr", labels=[0, 1, 2])
    except ValueError:
        auc = float("nan")
    return {
        "acc": accuracy_score(y, p),
        "macro_f1": f1_score(y, p, average="macro", labels=[0, 1, 2], zero_division=0),
        "qwk": cohen_kappa_score(y, p, weights="quadratic"),
        "auc_ovr": auc,
        "two_grade_errors": int(((y - p) ** 2 == 4).sum()),
        **{f"prec_g{i+1}": float(prec[i]) for i in range(3)},
        **{f"rec_g{i+1}": float(rec[i]) for i in range(3)},
    }

def train_fold(args, df, paths, fold, device):
    import torch
    from torch.utils.data import DataLoader, WeightedRandomSampler
    seed_everything(args.seed + fold)
    test = df[df.test_fold == fold]
    pool = df[df.test_fold != fold].reset_index(drop=True)
    # inner validation split (grouped by patient, stratified) used ONLY for early stopping
    sgk = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=args.seed)
    ytr = pool.grade.map(G2I).values
    tr_idx, va_idx = next(sgk.split(pool, ytr, pool.patient_id))
    train, val = pool.iloc[tr_idx], pool.iloc[va_idx]
    assert not set(train.patient_id) & set(val.patient_id)
    assert not (set(pool.patient_id) & set(test.patient_id))

    ttr, tev = build_transforms(args.img_size)
    # class-balanced sampling: each grade is drawn equally often per epoch (no copies saved to disk)
    y_tr = train.grade.map(G2I).values
    w = 1.0 / np.bincount(y_tr, minlength=3)[y_tr]
    sampler = WeightedRandomSampler(torch.tensor(w, dtype=torch.double), num_samples=len(train), replacement=True)
    kw = dict(num_workers=args.workers, pin_memory=(device.type == "cuda"))
    dl_tr = DataLoader(DS(train, paths, ttr), batch_size=args.bs, sampler=sampler, drop_last=True, **kw)
    dl_va = DataLoader(DS(val, paths, tev), batch_size=args.bs, shuffle=False, **kw)
    dl_te = DataLoader(DS(test, paths, tev), batch_size=args.bs, shuffle=False, **kw)

    model = make_model(args.model, not args.no_pretrained).to(device)
    head_names = {n for n, _ in model.get_classifier().named_parameters()}
    head = list(model.get_classifier().parameters())
    head_ids = {id(p) for p in head}
    backbone = [p for p in model.parameters() if id(p) not in head_ids]
    opt = torch.optim.AdamW([{"params": backbone, "lr": args.lr_backbone},
                             {"params": head, "lr": args.lr_head}], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    out_dir = Path(args.out) / args.model / f"fold{fold}"
    out_dir.mkdir(parents=True, exist_ok=True)
    best_f1, best_ep, bad, history = -1, 0, 0, []
    for ep in range(1, args.epochs + 1):
        for p in backbone:  # warm-up: train only the new head first
            p.requires_grad = ep > args.warmup
        tl, _, _, _ = run_epoch(model, dl_tr, device, loss_fn, opt, scaler)
        vl, vy, vp, vpr = run_epoch(model, dl_va, device, loss_fn)
        vf1 = f1_score(vy, vp, average="macro", labels=[0, 1, 2], zero_division=0)
        sched.step()
        history.append({"epoch": ep, "train_loss": tl, "val_loss": vl, "val_macro_f1": vf1})
        print(f"[{args.model} fold{fold}] ep{ep:02d} train_loss {tl:.3f} val_loss {vl:.3f} val_f1 {vf1:.3f}", flush=True)
        if vf1 > best_f1:
            best_f1, best_ep, bad = vf1, ep, 0
            torch.save(model.state_dict(), out_dir / "best.pt")
        else:
            bad += 1
            if ep > args.warmup and bad >= args.patience:
                print("early stop"); break
    pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)

    model.load_state_dict(torch.load(out_dir / "best.pt", map_location=device))
    _, ty, tp, tpr = run_epoch(model, dl_te, device, loss_fn)
    m = metrics(ty, tp, tpr)
    m.update({"fold": fold, "best_epoch": best_ep, "n_test": len(ty), "val_macro_f1": best_f1})
    pred = test.reset_index(drop=True)[["filename", "grade", "patient_id"]].copy()
    pred["pred"] = [GRADES[i] for i in tp]
    for i in range(3):
        pred[f"p_g{i+1}"] = tpr[:, i]
    pred.to_csv(out_dir / "test_predictions.csv", index=False)
    json.dump(m, open(out_dir / "test_metrics.json", "w"), indent=2)
    print(f"[{args.model} fold{fold}] TEST acc {m['acc']:.3f} macroF1 {m['macro_f1']:.3f} QWK {m['qwk']:.3f}")
    return m, pred


def preview_aug(args, df, paths):
    import torch
    from torchvision.utils import save_image
    ttr, tev = build_transforms(args.img_size)
    rows = []
    for g in GRADES:
        r = df[df.grade == g].iloc[0]
        img = Image.open(paths[r["filename"]]).convert("RGB")
        row = [tev(img)] + [ttr(img) for _ in range(5)]
        row = [(x * torch.tensor(STD)[:, None, None] + torch.tensor(MEAN)[:, None, None]).clamp(0, 1) for x in row]
        rows += row
    save_image(rows, "augmentation_preview.png", nrow=6, padding=4)
    print("saved augmentation_preview.png (left = original, then 5 random augmentations; one row per grade)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="resnet50", choices=["efficientnet_b4", "resnet50", "convnext_tiny"])
    ap.add_argument("--images", required=True, help="folder containing the dataset images (subfolders allowed)")
    ap.add_argument("--csv", default="labels_clinician_324.csv")
    ap.add_argument("--out", default="runs")
    ap.add_argument("--folds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    ap.add_argument("--img-size", type=int, default=320)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr-backbone", type=float, default=1e-4)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--check-data", action="store_true")
    ap.add_argument("--preview-aug", action="store_true")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    paths = index_images(args.images)
    missing = [f for f in df.filename if f not in paths]
    print(f"{len(df)} labelled rows, {len(paths)} images found on disk, {len(missing)} missing")
    if missing:
        print("missing examples:", missing[:10]); sys.exit(1)
    print(df.grade.value_counts().to_string())
    if args.check_data:
        return
    if args.preview_aug:
        return preview_aug(args, df, paths)

    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    all_m, all_pred = [], []
    for fold in args.folds:
        m, pred = train_fold(args, df, paths, fold, device)
        all_m.append(m); all_pred.append(pred)

    res = pd.DataFrame(all_m)
    cols = ["acc", "macro_f1", "qwk", "auc_ovr", "two_grade_errors"]
    summary = res[cols].agg(["mean", "std"])
    out = Path(args.out) / args.model
    res.to_csv(out / "fold_metrics.csv", index=False)
    summary.to_csv(out / "summary_mean_std.csv")
    oof = pd.concat(all_pred)
    cm = confusion_matrix(oof.grade, oof.pred, labels=GRADES)
    pd.DataFrame(cm, index=GRADES, columns=GRADES).to_csv(out / "confusion_matrix_pooled.csv")
    print("\n=== SUMMARY over folds", args.folds, "===")
    print(summary.round(3).to_string())
    print("pooled confusion matrix (rows = true):\n", cm)


if __name__ == "__main__":
    main()