import argparse
import random
import shutil
from pathlib import Path

from PIL import Image

from utils import ensure_dir, seed_everything

CLASSES = ["NORMAL", "PNEUMONIA"]


def valid_image(path: Path) -> bool:
    return path.suffix.lower() in {".jpg", ".jpeg", ".png"}


def preprocess_image(src: Path, dst: Path, img_size: int = 224) -> None:
    img = Image.open(src).convert("RGB").resize((img_size, img_size))
    img.save(dst)


def copy_split(files, output_root: Path, split: str, cls_name: str, img_size: int):
    split_dir = ensure_dir(output_root / split / cls_name)
    for fp in files:
        preprocess_image(fp, split_dir / fp.name, img_size=img_size)


def main():
    parser = argparse.ArgumentParser(description="Chest X-ray preprocessing and few-shot sampling")
    parser.add_argument("--input_dir", required=True, type=str)
    parser.add_argument("--output_dir", required=True, type=str)
    parser.add_argument("--per_class", type=int, default=100, help="few-shot samples per class")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    seed_everything(args.seed)
    in_root = Path(args.input_dir)
    out_root = Path(args.output_dir)

    if out_root.exists():
        shutil.rmtree(out_root)

    for cls_name in CLASSES:
        cls_files = [p for p in (in_root / "train" / cls_name).glob("*") if valid_image(p)]
        random.shuffle(cls_files)
        cls_files = cls_files[: min(args.per_class, len(cls_files))]

        n = len(cls_files)
        n_train = int(n * 0.7)
        n_val = int(n * 0.15)

        train_files = cls_files[:n_train]
        val_files = cls_files[n_train : n_train + n_val]
        test_files = [p for p in (in_root / "test" / cls_name).glob("*") if valid_image(p)]

        copy_split(train_files, out_root, "train", cls_name, args.img_size)
        copy_split(val_files, out_root, "val", cls_name, args.img_size)
        copy_split(test_files, out_root, "test", cls_name, args.img_size)

    print(f"Preprocessed dataset saved to: {out_root}")


if __name__ == "__main__":
    main()
