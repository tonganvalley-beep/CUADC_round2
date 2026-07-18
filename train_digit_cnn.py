"""Train and export the CUADC single-digit CNN without torchvision.

The first 700 labeled rows are the training set and the next 300 are the
held-out test set. Each rectified plate contributes a left and a right digit.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageOps
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


class DigitCNN(nn.Module):
    """Small (~25k parameter) classifier intended for Jetson TX2."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1, groups=16, bias=False),
            nn.Conv2d(32, 32, 1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1, groups=32, bias=False),
            nn.Conv2d(64, 64, 1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(64, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x).flatten(1))


def load_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = [r for r in csv.DictReader(handle)
                if r.get("status") == "labeled" and len(r.get("label", "")) == 2
                and r["label"].isdigit()]
    if not rows:
        raise ValueError("no valid labeled plates found")
    unique = []
    hashes: dict[str, str] = {}
    for row in rows:
        path = Path(row["path"])
        if not path.is_absolute():
            path = csv_path.parent / path
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest in hashes:
            if hashes[digest] != row["label"]:
                raise ValueError(f"conflicting labels for duplicate image: {path}")
            continue
        hashes[digest] = row["label"]
        unique.append(row)
    print(f"labeled_rows={len(rows)} unique_images={len(unique)} duplicates={len(rows)-len(unique)}")
    return unique


class DigitDataset(Dataset):
    def __init__(self, rows: list[dict[str, str]], csv_dir: Path, train: bool):
        self.samples = [(row, side) for row in rows for side in (0, 1)]
        self.csv_dir = csv_dir
        self.train = train

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row, side = self.samples[index]
        path = Path(row["path"])
        if not path.is_absolute():
            path = self.csv_dir / path
        gap = max(0, int(row.get("split_gap", 3)))
        with Image.open(path) as opened:
            image = ImageOps.exif_transpose(opened).convert("L")
        w, h = image.size
        mid = w // 2
        if side == 0:
            digit = image.crop((0, 0, max(1, mid - gap), h))
        else:
            digit = image.crop((min(w - 1, mid + gap), 0, w, h))
        digit = ImageOps.pad(digit, (32, 32), color=255, method=Image.Resampling.BILINEAR)
        if self.train:
            digit = self._augment(digit)
        array = np.asarray(digit, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).unsqueeze(0)
        tensor = (tensor - 0.5) / 0.5
        return tensor, int(row["label"][side])

    @staticmethod
    def _augment(image: Image.Image) -> Image.Image:
        angle = random.uniform(-5.0, 5.0)
        image = image.rotate(angle, resample=Image.Resampling.BILINEAR, fillcolor=255)
        image = ImageEnhance.Brightness(image).enhance(random.uniform(0.75, 1.25))
        image = ImageEnhance.Contrast(image).enhance(random.uniform(0.75, 1.35))
        return image


def class_weights(dataset: DigitDataset) -> torch.Tensor:
    counts = torch.zeros(10, dtype=torch.float32)
    for row, side in dataset.samples:
        counts[int(row["label"][side])] += 1
    return counts.sum() / counts.clamp_min(1.0)


@torch.no_grad()
def evaluate(model: nn.Module, rows: list[dict[str, str]], csv_dir: Path,
             device: torch.device, batch_size: int) -> dict[str, float]:
    loader = DataLoader(DigitDataset(rows, csv_dir, False), batch_size=batch_size,
                        shuffle=False, num_workers=0)
    predictions: list[int] = []
    labels: list[int] = []
    model.eval()
    for images, target in loader:
        pred = model(images.to(device)).argmax(1).cpu().tolist()
        predictions.extend(pred)
        labels.extend(target.tolist())
    left_ok = [predictions[i] == labels[i] for i in range(0, len(labels), 2)]
    right_ok = [predictions[i] == labels[i] for i in range(1, len(labels), 2)]
    plate_ok = [a and b for a, b in zip(left_ok, right_ok)]
    return {
        "left_accuracy": sum(left_ok) / len(left_ok),
        "right_accuracy": sum(right_ok) / len(right_ok),
        "plate_accuracy": sum(plate_ok) / len(plate_ok),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", default="digit_labels.csv")
    parser.add_argument("--output", default="weights/digit_cnn.ts")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--train-plates", type=int, default=600)
    parser.add_argument("--val-plates", type=int, default=60)
    args = parser.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    labels_path = Path(args.labels).resolve()
    rows = load_rows(labels_path)
    random.Random(args.seed).shuffle(rows)
    if args.train_plates <= 0 or args.train_plates >= len(rows):
        raise ValueError(f"--train-plates must be between 1 and {len(rows)-1}")
    development_rows, test_rows = rows[:args.train_plates], rows[args.train_plates:]
    if args.val_plates <= 0 or args.val_plates >= len(development_rows):
        raise ValueError("--val-plates must leave at least one actual training plate")
    train_rows = development_rows[:-args.val_plates]
    val_rows = development_rows[-args.val_plates:]
    train_set = DigitDataset(train_rows, labels_path.parent, True)
    weights = class_weights(train_set)
    loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                        num_workers=0, pin_memory=torch.cuda.is_available())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DigitCNN().to(device)
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    best_accuracy = -1.0
    best_state = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for images, target in loader:
            images, target = images.to(device), target.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), target)
            loss.backward(); optimizer.step()
            total_loss += float(loss.detach())
        scheduler.step()
        metrics = evaluate(model, val_rows, labels_path.parent, device, args.batch_size)
        print(f"epoch={epoch:03d} loss={total_loss/len(loader):.4f} "
              f"left={metrics['left_accuracy']:.4f} right={metrics['right_accuracy']:.4f} "
              f"plate={metrics['plate_accuracy']:.4f}")
        if metrics["plate_accuracy"] > best_accuracy:
            best_accuracy = metrics["plate_accuracy"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    assert best_state is not None
    model.load_state_dict(best_state); model.eval().cpu()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    scripted = torch.jit.script(model)
    scripted.save(str(output))
    torch.save(model.state_dict(), output.with_suffix(".pth"))
    metrics = evaluate(model, test_rows, labels_path.parent, torch.device("cpu"), args.batch_size)
    meta = {"input_size": [1, 32, 32], "normalization": "(x/255-0.5)/0.5",
            "train_plates": len(train_rows), "validation_plates": len(val_rows),
            "test_plates": len(test_rows), **metrics}
    output.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"saved {output}  metrics={metrics}")


if __name__ == "__main__":
    main()
