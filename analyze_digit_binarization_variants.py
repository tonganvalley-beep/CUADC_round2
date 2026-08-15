"""Generate visual comparisons for candidate shadow-robust binarization."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import torch

from digit_recognizer import DigitRecognizer


def read_image(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"无法读取 {path}")
    return cv2.resize(image, (100, 100), interpolation=cv2.INTER_LINEAR)


def write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix, image)
    if not ok:
        raise RuntimeError(f"无法编码 {path}")
    encoded.tofile(path)


def value_channel(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[:, :, 2]


def baseline_split_otsu(image: np.ndarray, split_gap: int = 3) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    middle = gray.shape[1] // 2
    gap = min(max(0, split_gap), max(0, middle - 1))
    result = np.full(gray.shape, 255, dtype=np.uint8)
    for start, end in ((0, middle - gap), (middle + gap, gray.shape[1])):
        region = cv2.GaussianBlur(gray[:, start:end], (3, 3), 0)
        _, result[:, start:end] = cv2.threshold(
            region, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
    return result


def adaptive_value(image: np.ndarray) -> np.ndarray:
    value = cv2.GaussianBlur(value_channel(image), (3, 3), 0)
    return cv2.adaptiveThreshold(
        value, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 7,
    )


def normalized_value(image: np.ndarray) -> np.ndarray:
    value = cv2.GaussianBlur(value_channel(image), (3, 3), 0)
    illumination = cv2.GaussianBlur(value, (0, 0), 13)
    normalized = cv2.divide(value, np.maximum(illumination, 1), scale=255)
    _, binary = cv2.threshold(
        normalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return binary


def find_split(binary: np.ndarray) -> int:
    """Find the inter-digit valley without assuming that it is x=50."""
    return DigitRecognizer.find_split(binary)


def component_split(binary: np.ndarray) -> int:
    """Estimate the split from the two most digit-like ink components."""
    ink = (binary < 128).astype(np.uint8)
    count, _labels, stats, centroids = cv2.connectedComponentsWithStats(ink, 8)
    candidates: list[tuple[float, int, int, int, int]] = []
    for index in range(1, count):
        x, y, width, height, area = (int(value) for value in stats[index])
        center_x, center_y = centroids[index]
        if area < 55 or width < 4 or height < 20:
            continue
        if width > 52 or height > 88:
            continue
        if not 8 <= center_y <= 92:
            continue
        # Prefer tall, substantial central components and penalize thin frame
        # fragments. A clipped right digit is allowed to touch the image edge.
        score = area + 2.0 * height - 0.6 * abs(center_x - 50)
        candidates.append((score, x, y, width, height))
    if len(candidates) < 2:
        return find_split(binary)
    chosen = sorted(candidates, reverse=True)[:2]
    chosen.sort(key=lambda item: item[1] + item[3] / 2.0)
    left, right = chosen
    left_edge = left[1] + left[3]
    right_edge = right[1]
    if right_edge > left_edge:
        split = (left_edge + right_edge) // 2
    else:
        split = round(((left[1] + left[3] / 2.0) +
                       (right[1] + right[3] / 2.0)) / 2.0)
    return max(30, min(70, int(split)))


@torch.inference_mode()
def predict_binary(recognizer: DigitRecognizer, binary: np.ndarray,
                   split_x: int, gap: int = 3) -> tuple[str, tuple[float, float]]:
    width = binary.shape[1]
    split_x = max(gap + 1, min(width - gap - 1, split_x))
    left = binary[:, :split_x - gap]
    right = binary[:, split_x + gap:]
    batch = np.stack([recognizer._prepare(left), recognizer._prepare(right)])[:, None]
    logits = recognizer.model(torch.from_numpy(batch).to(recognizer.device))
    probabilities = torch.softmax(logits.float(), dim=1)
    confidences, digits = probabilities.max(dim=1)
    return (f"{int(digits[0])}{int(digits[1])}",
            (float(confidences[0]), float(confidences[1])))


def labeled_tile(image: np.ndarray, label: str, scale: int = 2) -> np.ndarray:
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    shown = cv2.resize(image, (100 * scale, 100 * scale),
                       interpolation=cv2.INTER_NEAREST)
    result = np.full((shown.shape[0] + 30, shown.shape[1], 3), 245, np.uint8)
    result[30:] = shown
    cv2.putText(result, label, (5, 21), cv2.FONT_HERSHEY_SIMPLEX,
                0.52, (20, 20, 20), 1, cv2.LINE_AA)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--review", type=Path,
                        default=Path("digit_review_pic22_23/review.csv"))
    parser.add_argument("--output", type=Path,
                        default=Path("digit_review_pic22_23/variant_analysis"))
    parser.add_argument("--model", type=Path,
                        default=Path("weights/digit_cnn_v5_pic12_21_split_otsu_finetune.ts"))
    args = parser.parse_args()
    review_path = args.review.resolve()
    with review_path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))

    recognizer = DigitRecognizer(str(args.model), device="cpu", confidence=0.0)
    sheets: dict[str, list[np.ndarray]] = {}
    prediction_rows: list[dict[str, str | int | float]] = []
    for row in rows:
        source = (review_path.parent / row["path"]).resolve()
        image = read_image(source)
        variants = {
            "baseline": baseline_split_otsu(image, int(row["split_gap"])),
            "adaptive_v": adaptive_value(image),
            "normalized_v": normalized_value(image),
        }
        for name, binary in variants.items():
            for split_name, split_x in (
                ("fixed", 50),
                ("valley", find_split(binary)),
                ("component", component_split(binary)),
            ):
                predicted, confidence = predict_binary(recognizer, binary, split_x)
                prediction_rows.append({
                    "path": row["path"], "label": row["correct_label"],
                    "variant": name, "split": split_name, "split_x": split_x,
                    "predicted": predicted,
                    "correct": int(predicted == row["correct_label"]),
                    "left_confidence": confidence[0],
                    "right_confidence": confidence[1],
                })
        dataset = source.parent.parent.name
        title = f"{dataset}/{source.name} true={row['correct_label']} bad={row['binary_status'] == 'unreasonable'}"
        tiles = [labeled_tile(image, title)]
        tiles.extend(labeled_tile(value, name) for name, value in variants.items())
        sheets.setdefault(dataset, []).append(cv2.hconcat(tiles))
        for name, value in variants.items():
            write_image(args.output / name / f"{dataset}_{source.stem}.png", value)

    for dataset, row_images in sheets.items():
        write_image(args.output / f"{dataset}_comparison.jpg", cv2.vconcat(row_images))
    with (args.output / "predictions.csv").open(
            "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=prediction_rows[0].keys())
        writer.writeheader()
        writer.writerows(prediction_rows)
    with (args.output / "normalized_training_labels.csv").open(
            "w", newline="", encoding="utf-8-sig") as handle:
        fields = ("path", "label", "status", "split_gap", "split_x")
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            source = (review_path.parent / row["path"]).resolve()
            dataset = source.parent.parent.name
            binary_name = f"{dataset}_{source.stem}.png"
            binary = cv2.imdecode(
                np.fromfile(args.output / "normalized_v" / binary_name, dtype=np.uint8),
                cv2.IMREAD_GRAYSCALE,
            )
            writer.writerow({
                "path": f"normalized_v/{binary_name}",
                "label": row["correct_label"],
                "status": "labeled",
                "split_gap": 3,
                "split_x": find_split(binary),
            })
    for variant in ("baseline", "adaptive_v", "normalized_v"):
        for split_name in ("fixed", "valley", "component"):
            group = [item for item in prediction_rows
                     if item["variant"] == variant and item["split"] == split_name]
            correct = sum(int(item["correct"]) for item in group)
            print(f"{variant:12s} {split_name:6s} accuracy={correct}/{len(group)}")
    print(f"rows={len(rows)} output={args.output.resolve()}")


if __name__ == "__main__":
    main()
