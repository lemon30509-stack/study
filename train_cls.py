import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.utils.class_weight import compute_class_weight
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import ConcatDataset, DataLoader
from torchvision import datasets, models, transforms

from config import CLSConfig
from utils import cls_metrics, ensure_dir, save_json, seed_everything


def build_model():
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    in_f = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_f, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
        nn.Linear(256, 2),
    )
    return model


def evaluate(model, loader, device):
    model.eval()
    ys, ps, probs = [], [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            prob = torch.softmax(logits, dim=1)[:, 1]
            pred = torch.argmax(logits, dim=1)
            ys.extend(y.cpu().tolist())
            ps.extend(pred.cpu().tolist())
            probs.extend(prob.cpu().tolist())
    return cls_metrics(ys, ps, probs)


def build_train_dataset(train_dir: str, synthetic_dir: str | None, transform):
    base = datasets.ImageFolder(train_dir, transform=transform)
    if not synthetic_dir:
        return base

    syn_path = Path(synthetic_dir)
    if syn_path.exists():
        synthetic = datasets.ImageFolder(str(syn_path), transform=transform)
        if synthetic.class_to_idx != base.class_to_idx:
            raise ValueError(
                f"synthetic_dir classes {synthetic.class_to_idx} do not match train classes {base.class_to_idx}"
            )
        return ConcatDataset([base, synthetic])

    print(f"[Warn] synthetic_dir not found, fallback to original train set: {synthetic_dir}")
    return base


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", required=True)
    parser.add_argument("--val_dir", required=True)
    parser.add_argument("--test_dir", required=True)
    parser.add_argument("--synthetic_dir", default=None, help="optional GAN synthetic images root")
    parser.add_argument("--output_dir", default="outputs/classifier")
    args = parser.parse_args()

    cfg = CLSConfig()
    seed_everything(42)
    out = ensure_dir(args.output_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_tfm = transforms.Compose([
        transforms.Resize((cfg.img_size, cfg.img_size)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomResizedCrop(cfg.img_size, scale=(0.8, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    test_tfm = transforms.Compose([
        transforms.Resize((cfg.img_size, cfg.img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    raw_train_ds = datasets.ImageFolder(args.train_dir, transform=train_tfm)
    train_ds = build_train_dataset(args.train_dir, args.synthetic_dir, train_tfm)
    val_ds = datasets.ImageFolder(args.val_dir, transform=test_tfm)
    test_ds = datasets.ImageFolder(args.test_dir, transform=test_tfm)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=2)

    y = np.array(raw_train_ds.targets)
    w = compute_class_weight(class_weight="balanced", classes=np.array([0, 1]), y=y)

    model = build_model().to(device)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32, device=device))

    backbone_params, head_params = [], []
    for n, p in model.named_parameters():
        (head_params if n.startswith("fc") else backbone_params).append(p)

    opt = AdamW(
        [{"params": backbone_params, "lr": cfg.lr_backbone}, {"params": head_params, "lr": cfg.lr_head}],
        weight_decay=cfg.weight_decay,
    )
    scheduler = CosineAnnealingLR(opt, T_max=50, eta_min=1e-6)

    best_f1 = -1
    stale = 0
    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        for x, yb in train_loader:
            x, yb = x.to(device), yb.to(device)
            logits = model(x)
            loss = criterion(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()

        scheduler.step()
        val_m = evaluate(model, val_loader, device)
        print(f"Epoch {epoch}/{cfg.max_epochs} val_f1={val_m['f1']:.4f} val_auc={val_m['auc']:.4f}")

        if val_m["f1"] > best_f1:
            best_f1 = val_m["f1"]
            stale = 0
            torch.save(model.state_dict(), out / "best_classifier.pth")
        else:
            stale += 1
            if stale >= cfg.patience:
                print("Early stopping triggered.")
                break

    model.load_state_dict(torch.load(out / "best_classifier.pth", map_location=device))
    test_m = evaluate(model, test_loader, device)
    print("Test:", test_m)
    save_json(test_m, out / "metrics.json")


if __name__ == "__main__":
    main()
