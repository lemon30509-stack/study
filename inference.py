import argparse

import torch
from PIL import Image
from torchvision import models, transforms


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
    parser.add_argument("--image", required=True)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model().to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()

    tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    x = tfm(Image.open(args.image).convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        prob = torch.softmax(logits, dim=1)[0]
        pred = int(torch.argmax(prob).item())

    print({"label": CLASSES[pred], "confidence": float(prob[pred].item())})


if __name__ == "__main__":
    main()
