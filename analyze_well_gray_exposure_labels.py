"""Analyze labeled grayscale exposure crops for detect_well_exposure(3).py."""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path
from typing import Callable

import cv2
import numpy as np


FEATURE_FIELDS = (
    "mean",
    "std",
    "p05",
    "p10",
    "p25",
    "p50",
    "p75",
    "p90",
    "p95",
    "p99",
    "contrast_90_10",
    "center_mean",
    "center_std",
    "center_p10",
    "center_p25",
    "center_p50",
    "center_p75",
    "center_p90",
    "center_p95",
    "center_contrast_90_10",
    "center_minus_border_mean",
    "bright_ratio_180",
    "bright_ratio_200",
    "bright_ratio_210",
    "bright_ratio_220",
    "bright_ratio_230",
    "bright_ratio_240",
    "bright_ratio_245",
    "bright_ratio_250",
    "center_bright_ratio_180",
    "center_bright_ratio_200",
    "center_bright_ratio_210",
    "center_bright_ratio_220",
    "center_bright_ratio_230",
    "center_bright_ratio_240",
    "center_bright_ratio_245",
    "center_bright_ratio_250",
)


def read_image(path: Path) -> np.ndarray:
    encoded = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError("cannot read image: {}".format(path))
    return image


def center_and_border(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    height, width = gray.shape
    y1, y2 = int(round(height * 0.20)), int(round(height * 0.80))
    x1, x2 = int(round(width * 0.20)), int(round(width * 0.80))
    center = gray[y1:y2, x1:x2]
    border_mask = np.ones(gray.shape, dtype=bool)
    border_mask[y1:y2, x1:x2] = False
    return center, gray[border_mask]


def metrics(gray: np.ndarray) -> dict[str, float]:
    center, border = center_and_border(gray)
    p05, p10, p25, p50, p75, p90, p95, p99 = np.percentile(
        gray, (5, 10, 25, 50, 75, 90, 95, 99)
    )
    cp10, cp25, cp50, cp75, cp90, cp95 = np.percentile(
        center, (10, 25, 50, 75, 90, 95)
    )
    result = {
        "mean": float(gray.mean()),
        "std": float(gray.std()),
        "p05": float(p05),
        "p10": float(p10),
        "p25": float(p25),
        "p50": float(p50),
        "p75": float(p75),
        "p90": float(p90),
        "p95": float(p95),
        "p99": float(p99),
        "contrast_90_10": float(p90 - p10),
        "center_mean": float(center.mean()),
        "center_std": float(center.std()),
        "center_p10": float(cp10),
        "center_p25": float(cp25),
        "center_p50": float(cp50),
        "center_p75": float(cp75),
        "center_p90": float(cp90),
        "center_p95": float(cp95),
        "center_contrast_90_10": float(cp90 - cp10),
        "center_minus_border_mean": float(center.mean() - border.mean()),
    }
    for threshold in (180, 200, 210, 220, 230, 240, 245, 250):
        result["bright_ratio_{}".format(threshold)] = float(
            np.mean(gray >= threshold)
        )
        result["center_bright_ratio_{}".format(threshold)] = float(
            np.mean(center >= threshold)
        )
    return result


def scores(truth: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    tp = int(np.sum(predicted & truth))
    tn = int(np.sum(~predicted & ~truth))
    fp = int(np.sum(predicted & ~truth))
    fn = int(np.sum(~predicted & truth))
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    precision = tp / max(tp + fp, 1)
    return {
        "balanced_accuracy": (recall + specificity) / 2.0,
        "accuracy": (tp + tn) / max(len(truth), 1),
        "recall": recall,
        "specificity": specificity,
        "precision": precision,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def candidate_thresholds(values: np.ndarray) -> np.ndarray:
    unique = np.unique(values)
    if len(unique) == 1:
        return unique
    return (unique[:-1] + unique[1:]) / 2.0


def best_stump(values: np.ndarray, truth: np.ndarray) -> dict[str, object]:
    best: dict[str, object] | None = None
    for threshold in candidate_thresholds(values):
        for direction, predicate in (
            (">=", lambda x, t: x >= t),
            ("<=", lambda x, t: x <= t),
        ):
            result: dict[str, object] = {
                "threshold": float(threshold),
                "direction": direction,
            }
            result.update(scores(truth, predicate(values, threshold)))
            ordering = (
                float(result["balanced_accuracy"]),
                float(result["recall"]),
                float(result["specificity"]),
            )
            if best is None or ordering > (
                float(best["balanced_accuracy"]),
                float(best["recall"]),
                float(best["specificity"]),
            ):
                best = result
    assert best is not None
    return best


def condition(direction: str) -> Callable[[np.ndarray, float], np.ndarray]:
    return (lambda x, t: x >= t) if direction == ">=" else (lambda x, t: x <= t)


def quantile_text(values: np.ndarray) -> str:
    q = np.percentile(values, (0, 10, 25, 50, 75, 90, 100))
    return "/".join("{:.3f}".format(value) for value in q)


def load_rows(labels_path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with labels_path.open("r", newline="", encoding="utf-8-sig") as handle:
        for label_row in csv.DictReader(handle):
            if label_row.get("status") != "labeled":
                continue
            label = label_row.get("exposure_label")
            if label not in ("OVEREXPOSED", "NORMAL"):
                continue
            image_path = Path(label_row["source"])
            if not image_path.is_absolute():
                image_path = labels_path.parent / image_path
            row: dict[str, object] = {
                "source": label_row["source"],
                "dataset": label_row["dataset"],
                "filename": label_row["filename"],
                "exposure_label": label,
                "content_hash": hashlib.sha256(image_path.read_bytes()).hexdigest(),
            }
            row.update(metrics(read_image(image_path)))
            rows.append(row)
    return rows


def deduplicate_rows(
    rows: list[dict[str, object]],
) -> tuple[list[dict[str, object]], list[list[dict[str, object]]]]:
    groups: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        groups.setdefault(str(row["content_hash"]), []).append(row)
    unique_rows = []
    conflicts = []
    for group in groups.values():
        labels = {str(row["exposure_label"]) for row in group}
        if len(labels) > 1:
            conflicts.append(group)
        else:
            unique_rows.append(group[0])
    return unique_rows, conflicts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("labels", type=Path)
    parser.add_argument(
        "--output", type=Path, default=Path("well_exposure_gray_features.csv")
    )
    parser.add_argument("--threshold", type=float, default=203.0)
    args = parser.parse_args()
    labels_path = args.labels.resolve()
    rows = load_rows(labels_path)
    if not rows:
        raise SystemExit("no labeled NORMAL/OVEREXPOSED rows")

    for row in rows:
        predicted = (
            "OVEREXPOSED"
            if float(row["center_p10"]) >= args.threshold
            else "NORMAL"
        )
        row["program_state"] = predicted
        row["match"] = str(predicted == row["exposure_label"])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8-sig") as handle:
        fields = (
            "source", "dataset", "filename", "exposure_label", "program_state",
            "match", "content_hash"
        ) + FEATURE_FIELDS
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    unique_rows, conflicts = deduplicate_rows(rows)
    truth = np.asarray(
        [row["exposure_label"] == "OVEREXPOSED" for row in unique_rows]
    )
    print(
        "labeled={} over={} normal={}".format(
            len(rows),
            sum(row["exposure_label"] == "OVEREXPOSED" for row in rows),
            sum(row["exposure_label"] == "NORMAL" for row in rows),
        )
    )
    print(
        "unique_consistent={} over={} normal={} conflicting_groups={}".format(
            len(unique_rows), int(truth.sum()), int((~truth).sum()), len(conflicts)
        )
    )
    for group in conflicts:
        print(
            "CONFLICT "
            + " | ".join(
                "{}={}".format(row["source"], row["exposure_label"]) for row in group
            )
        )

    full_truth = np.asarray(
        [row["exposure_label"] == "OVEREXPOSED" for row in rows]
    )
    full_predicted = np.asarray(
        [float(row["center_p10"]) >= args.threshold for row in rows]
    )
    unique_predicted = np.asarray(
        [float(row["center_p10"]) >= args.threshold for row in unique_rows]
    )
    print("\nSelected rule: center_p10 >= {:.1f}".format(args.threshold))
    print("all labeled: {}".format(scores(full_truth, full_predicted)))
    print("unique consistent: {}".format(scores(truth, unique_predicted)))

    ranked = []
    for feature in FEATURE_FIELDS:
        values = np.asarray([float(row[feature]) for row in unique_rows])
        result = best_stump(values, truth)
        result["feature"] = feature
        ranked.append(result)
    ranked.sort(
        key=lambda row: (
            float(row["balanced_accuracy"]),
            float(row["recall"]),
            float(row["specificity"]),
        ),
        reverse=True,
    )

    print("\nTop single-threshold rules (fit on all labeled samples):")
    for row in ranked[:15]:
        print(
            "{feature:30s} {direction} {threshold:8.3f}  "
            "BA={balanced_accuracy:.3f} acc={accuracy:.3f} "
            "rec={recall:.3f} spec={specificity:.3f} "
            "TP={tp} TN={tn} FP={fp} FN={fn}".format(**row)
        )

    print("\nClass quantiles: min/p10/p25/p50/p75/p90/max")
    for row in ranked[:10]:
        feature = str(row["feature"])
        values = np.asarray([float(item[feature]) for item in unique_rows])
        print(
            "{:30s} OVER {} | NORMAL {}".format(
                feature, quantile_text(values[truth]), quantile_text(values[~truth])
            )
        )

    print("\nwrote {}".format(args.output))


if __name__ == "__main__":
    main()
