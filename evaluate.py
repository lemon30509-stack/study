import argparse

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

from utils import cls_metrics, ensure_dir, save_json


CLASSES = ["NORMAL", "PNEUMONIA"]


def build_model():
    model = models.resnet18(weights=None)
    in_f = model.fc.in_features
    model.fc = torch.nn.Sequential(
        torch.nn.Linear(in_f, 256),
        torch.nn.ReLU(inplace=True),
        torch.nn.Dropout(0.5),
        torch.nn.Linear(256, 2),
    )
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", default="outputs/eval")
    args = parser.parse_args()

    out = ensure_dir(args.output_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    ds = datasets.ImageFolder(args.data_dir, transform=tfm)
    loader = DataLoader(ds, batch_size=32, shuffle=False)

    model = build_model().to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()

    y_true, y_pred, y_prob = [], [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            prob = torch.softmax(logits, dim=1)[:, 1].cpu().tolist()
            pred = torch.argmax(logits, dim=1).cpu().tolist()
            y_true.extend(y.tolist())
            y_pred.extend(pred)
            y_prob.extend(prob)

    m = cls_metrics(y_true, y_pred, y_prob)
    save_json(m, out / "metrics.json")
    print(m)

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASSES, yticklabels=CLASSES)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(out / "confusion_matrix.png", dpi=150)


if __name__ == "__main__":
    main()
