"""Train and export the CUADC single-digit CNN without torchvision.

The first 700 labeled rows are the training set and the next 300 are the
held-out test set. Each rectified plate contributes a left and a right digit.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import torch
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader, Dataset


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


def load_single_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = [r for r in csv.DictReader(handle)
                if r.get("status") == "saved"
                and len(r.get("label", "")) == 1
                and r["label"].isdigit()]
    unique = []
    hashes: dict[str, str] = {}
    for row in rows:
        path = resolve_single_digit_path(row, csv_path.parent)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest in hashes:
            if hashes[digest] != row["label"]:
                raise ValueError(f"conflicting single-digit labels: {path}")
            continue
        hashes[digest] = row["label"]
        unique.append(row)
    print(f"single_rows={len(rows)} single_unique={len(unique)} "
          f"single_duplicates={len(rows)-len(unique)}")
    return unique


def split_extra_rows(rows: list[dict[str, str]], validation_fraction: float,
                     seed: int) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Keep rare labels in training and hold out repeated labels for validation."""
    if not 0.0 <= validation_fraction < 1.0:
        raise ValueError("--extra-val-fraction must be in [0, 1)")
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["label"]].append(row)
    train_rows: list[dict[str, str]] = []
    val_rows: list[dict[str, str]] = []
    for label in sorted(grouped):
        group = grouped[label]
        random.Random(seed + int(label)).shuffle(group)
        val_count = 0
        if len(group) >= 3 and validation_fraction > 0:
            val_count = min(len(group) - 1,
                            max(1, round(len(group) * validation_fraction)))
        val_rows.extend(group[:val_count])
        train_rows.extend(group[val_count:])
    return train_rows, val_rows


def resolve_single_digit_path(row: dict[str, str], csv_dir: Path) -> Path:
    """Resolve old absolute CSV paths after the project directory is moved."""
    path = Path(row["digit_path"])
    if not path.is_absolute():
        return csv_dir / path
    if path.exists():
        return path

    parts = path.parts
    lowered = [part.lower() for part in parts]
    if "single_digit_dataset" in lowered:
        index = lowered.index("single_digit_dataset")
        relocated = csv_dir.joinpath(*parts[index:])
        if relocated.exists():
            return relocated

    relocated = (csv_dir / "single_digit_dataset" / "digits_32" /
                 row["label"] / path.name)
    if relocated.exists():
        return relocated
    return path


class DigitDataset(Dataset):
    def __init__(self, rows: list[dict[str, str]], csv_dir: Path, train: bool):
        self.samples = [(row, side) for row in rows for side in (0, 1)]
        self.csv_dir = csv_dir
        self.train = train
        self.targets = [int(row["label"][side]) for row, side in self.samples]

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
        if self.train:
            digit = self._morphology(digit)
        digit = ImageOps.pad(digit, (32, 32), color=255, method=Image.Resampling.BILINEAR)
        if self.train:
            digit = self._augment(digit)
            digit = self._degrade(digit)
        array = np.asarray(digit, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).unsqueeze(0)
        tensor = (tensor - 0.5) / 0.5
        return tensor, int(row["label"][side])

    @staticmethod
    def _morphology(image: Image.Image) -> Image.Image:
        # The source plates are binary (black digits on a white background).
        # Morphology is applied before the 32x32 resize: a 3x3 kernel at the
        # source resolution is mild enough to imitate thresholding errors.
        # Pillow's MinFilter expands black foreground; MaxFilter thins it.
        operation = random.random()
        if operation < 0.10:
            image = image.filter(ImageFilter.MinFilter(3))  # black dilation
        elif operation < 0.20:
            image = image.filter(ImageFilter.MaxFilter(3))  # black erosion
        elif operation < 0.25:
            image = image.filter(ImageFilter.MaxFilter(3)).filter(
                ImageFilter.MinFilter(3))  # black opening
        elif operation < 0.30:
            image = image.filter(ImageFilter.MinFilter(3)).filter(
                ImageFilter.MaxFilter(3))  # black closing
        return image

    @staticmethod
    def _augment(image: Image.Image) -> Image.Image:
        angle = random.uniform(-5.0, 5.0)
        image = image.rotate(angle, resample=Image.Resampling.BILINEAR, fillcolor=255)
        image = ImageEnhance.Brightness(image).enhance(random.uniform(0.75, 1.25))
        image = ImageEnhance.Contrast(image).enhance(random.uniform(0.75, 1.35))
        return image

    @staticmethod
    def _degrade(image: Image.Image) -> Image.Image:
        """Create one mild camera/threshold degradation for training only."""
        operation = random.random()
        if operation < 0.20:
            side = random.randint(20, 28)
            image = image.resize((side, side), Image.Resampling.BILINEAR)
            image = image.resize((32, 32), Image.Resampling.BILINEAR)
        elif operation < 0.35:
            image = image.filter(ImageFilter.GaussianBlur(
                radius=random.uniform(0.3, 0.8)))
        elif operation < 0.45:
            array = np.asarray(image, dtype=np.uint8).copy()
            count = random.randint(2, 8)
            ys = np.random.randint(0, array.shape[0], size=count)
            xs = np.random.randint(0, array.shape[1], size=count)
            values = np.random.choice(
                np.array([0, 255], dtype=np.uint8), size=count)
            array[ys, xs] = values
            image = Image.fromarray(array, mode="L")
        return image


class SingleDigitDataset(Dataset):
    """Already rectified 32x32 single digits produced by the corner tool."""

    def __init__(self, rows: list[dict[str, str]], csv_dir: Path, train: bool):
        self.rows = rows
        self.csv_dir = csv_dir
        self.train = train
        self.targets = [int(row["label"]) for row in rows]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.rows[index]
        path = resolve_single_digit_path(row, self.csv_dir)
        with Image.open(path) as opened:
            digit = ImageOps.exif_transpose(opened).convert("L")
        digit = ImageOps.pad(
            digit, (32, 32), color=255, method=Image.Resampling.BILINEAR)
        if self.train:
            digit = DigitDataset._augment(digit)
            digit = DigitDataset._degrade(digit)
        array = np.asarray(digit, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).unsqueeze(0)
        tensor = (tensor - 0.5) / 0.5
        return tensor, int(row["label"])


class RepeatedDigitSubset(Dataset):
    """Repeat selected digit samples while preserving online augmentation."""

    def __init__(self, dataset: DigitDataset, indices: list[int]):
        self.dataset = dataset
        self.indices = indices
        self.targets = [dataset.targets[index] for index in indices]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        return self.dataset[self.indices[index]]


def balance_extra_digits(dataset: DigitDataset, base_targets: list[int],
                         target: int, max_repeat: int) -> tuple[RepeatedDigitSubset, dict[int, int]]:
    """Repeat extra digit crops toward a target derived from the base set."""
    base_counts = [base_targets.count(digit) for digit in range(10)]
    extra_counts = [dataset.targets.count(digit) for digit in range(10)]
    if target <= 0:
        target = max(base_counts)
    repeats: dict[int, int] = {}
    for digit in range(10):
        if extra_counts[digit] == 0:
            repeats[digit] = 0
            continue
        needed = max(0, target - base_counts[digit])
        repeats[digit] = min(max_repeat, max(1, math.ceil(needed / extra_counts[digit])))
    indices = [index for index, digit in enumerate(dataset.targets)
               for _ in range(repeats[digit])]
    return RepeatedDigitSubset(dataset, indices), repeats

def class_weights(datasets) -> torch.Tensor:
    counts = torch.zeros(10, dtype=torch.float32)
    for dataset in datasets:
        for target in dataset.targets:
            counts[target] += 1
    return counts.sum() / counts.clamp_min(1.0)


@torch.no_grad()
def evaluate_single(model: nn.Module, rows: list[dict[str, str]],
                    csv_dir: Path, device: torch.device,
                    batch_size: int) -> float:
    if not rows:
        return float("nan")
    loader = DataLoader(
        SingleDigitDataset(rows, csv_dir, False), batch_size=batch_size,
        shuffle=False, num_workers=0)
    correct = total = 0
    model.eval()
    for images, target in loader:
        prediction = model(images.to(device)).argmax(1).cpu()
        correct += int((prediction == target).sum())
        total += len(target)
    return correct / total


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
    parser.add_argument("--extra-labels", default="",
                        help="additional labeled two-digit CSV used for fine-tuning")
    parser.add_argument("--extra-val-fraction", type=float, default=0.20,
                        help="per-label validation fraction for --extra-labels")
    parser.add_argument("--extra-repeat", type=int, default=1,
                        help="repeat extra training samples per epoch")
    parser.add_argument("--balance-extra-to", type=int, default=-1,
                        help="balance digits found in extra data to this base count; "
                             "0 uses the largest base class, -1 disables balancing")
    parser.add_argument("--max-extra-repeat", type=int, default=8,
                        help="maximum online repeat factor for one extra digit")
    parser.add_argument("--single-labels", default="",
                        help="CSV from annotate_single_digit_corners.py")
    parser.add_argument("--single-max-per-class", type=int, default=0,
                        help="cap single-digit training samples per class; 0 keeps all")
    parser.add_argument("--output", default="weights/digit_cnn.ts")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--init-weights", default="",
                        help="optional state_dict .pth used to initialize training")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--train-plates", type=int, default=600)
    parser.add_argument("--val-plates", type=int, default=60)
    args = parser.parse_args()

    if args.extra_repeat < 1:
        raise ValueError("--extra-repeat must be at least 1")
    if args.balance_extra_to < -1:
        raise ValueError("--balance-extra-to must be -1, 0, or a positive count")
    if args.max_extra_repeat < 1:
        raise ValueError("--max-extra-repeat must be at least 1")

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
    training_sets = [train_set]
    extra_train_rows = []
    extra_val_rows = []
    extra_digit_repeats: dict[int, int] = {}
    extra_csv_dir = labels_path.parent
    if args.extra_labels:
        extra_path = Path(args.extra_labels).resolve()
        extra_rows = load_rows(extra_path)
        extra_train_rows, extra_val_rows = split_extra_rows(
            extra_rows, args.extra_val_fraction, args.seed)
        extra_csv_dir = extra_path.parent
        if extra_train_rows:
            extra_set = DigitDataset(extra_train_rows, extra_csv_dir, True)
            if args.balance_extra_to >= 0:
                balanced_set, extra_digit_repeats = balance_extra_digits(
                    extra_set, train_set.targets, args.balance_extra_to,
                    args.max_extra_repeat)
                training_sets.extend([balanced_set] * args.extra_repeat)
            else:
                training_sets.extend([extra_set] * args.extra_repeat)
        print(f"extra_split train={len(extra_train_rows)} "
              f"val={len(extra_val_rows)} repeat={args.extra_repeat}")
        if extra_digit_repeats:
            print("extra_digit_repeats=" + json.dumps(extra_digit_repeats, sort_keys=True))
    single_val_rows = []
    single_test_rows = []
    single_csv_dir = labels_path.parent
    if args.single_labels:
        single_path = Path(args.single_labels).resolve()
        single_rows = load_single_rows(single_path)
        single_train_rows = [
            row for row in single_rows
            if row.get("split", "train") in ("", "train")]
        single_val_rows = [
            row for row in single_rows if row.get("split") == "val"]
        single_test_rows = [
            row for row in single_rows if row.get("split") == "test"]
        if args.single_max_per_class > 0:
            capped_rows = []
            for digit in range(10):
                digit_rows = [row for row in single_train_rows
                              if int(row["label"]) == digit]
                random.Random(args.seed + digit).shuffle(digit_rows)
                capped_rows.extend(digit_rows[:args.single_max_per_class])
            single_train_rows = capped_rows
        single_csv_dir = single_path.parent
        if single_train_rows:
            training_sets.append(
                SingleDigitDataset(single_train_rows, single_csv_dir, True))
        print(f"single_split train={len(single_train_rows)} "
              f"val={len(single_val_rows)} test={len(single_test_rows)}")
    weights = class_weights(training_sets)
    combined_train = (training_sets[0] if len(training_sets) == 1
                      else ConcatDataset(training_sets))
    loader = DataLoader(combined_train, batch_size=args.batch_size, shuffle=True,
                        num_workers=0, pin_memory=torch.cuda.is_available())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DigitCNN().to(device)
    if args.init_weights:
        initial_state = torch.load(args.init_weights, map_location="cpu")
        model.load_state_dict(initial_state, strict=True)
        print(f"initialized_from={Path(args.init_weights).resolve()}")
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
        extra_val_metrics = (evaluate(
            model, extra_val_rows, extra_csv_dir, device, args.batch_size)
            if extra_val_rows else None)
        single_val_accuracy = evaluate_single(
            model, single_val_rows, single_csv_dir, device, args.batch_size)
        print(f"epoch={epoch:03d} loss={total_loss/len(loader):.4f} "
              f"left={metrics['left_accuracy']:.4f} right={metrics['right_accuracy']:.4f} "
              f"plate={metrics['plate_accuracy']:.4f} "
              f"extra_plate={extra_val_metrics['plate_accuracy'] if extra_val_metrics else float('nan'):.4f} "
              f"single_val={single_val_accuracy:.4f}")
        selection_accuracy = metrics["plate_accuracy"]
        if extra_val_metrics:
            selection_accuracy = (
                0.7 * metrics["plate_accuracy"]
                + 0.3 * extra_val_metrics["plate_accuracy"])
        elif single_val_rows:
            selection_accuracy = (
                0.7 * metrics["plate_accuracy"] + 0.3 * single_val_accuracy)
        if selection_accuracy > best_accuracy:
            best_accuracy = selection_accuracy
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    assert best_state is not None
    model.load_state_dict(best_state); model.eval().cpu()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    scripted = torch.jit.script(model)
    scripted.save(str(output))
    torch.save(model.state_dict(), output.with_suffix(".pth"))
    metrics = evaluate(model, test_rows, labels_path.parent, torch.device("cpu"), args.batch_size)
    extra_val_metrics = (evaluate(
        model, extra_val_rows, extra_csv_dir, torch.device("cpu"), args.batch_size)
        if extra_val_rows else None)
    single_test_accuracy = evaluate_single(
        model, single_test_rows, single_csv_dir, torch.device("cpu"),
        args.batch_size)
    meta = {"input_size": [1, 32, 32], "normalization": "(x/255-0.5)/0.5",
            "train_plates": len(train_rows), "validation_plates": len(val_rows),
            "test_plates": len(test_rows),
            "extra_train_plates": len(extra_train_rows),
            "extra_validation_plates": len(extra_val_rows),
            "extra_repeat": args.extra_repeat,
            "extra_digit_repeats": extra_digit_repeats,
            "extra_validation_metrics": extra_val_metrics,
            "single_validation_samples": len(single_val_rows),
            "single_test_samples": len(single_test_rows),
            "single_test_accuracy": (
                None if math.isnan(single_test_accuracy)
                else single_test_accuracy), **metrics}
    output.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"saved {output}  metrics={metrics}")


if __name__ == "__main__":
    main()
