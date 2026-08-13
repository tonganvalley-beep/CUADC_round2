"""Summarize manual blur labels and image-quality metrics."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


METRICS = ("mean_gray", "laplacian_variance", "black_ratio", "otsu_threshold")


def quantiles(values):
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "q25": float(np.percentile(array, 25)),
        "q75": float(np.percentile(array, 75)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def best_threshold(rows, metric):
    values = np.asarray([float(row[metric]) for row in rows], dtype=np.float64)
    target = np.asarray([row["blur_label"] == "blurred" for row in rows])
    unique = np.unique(values)
    thresholds = np.r_[unique[0] - 1e-9, (unique[:-1] + unique[1:]) / 2,
                       unique[-1] + 1e-9]
    best = None
    for direction in ("below", "above"):
        for threshold in thresholds:
            prediction = values <= threshold if direction == "below" else values >= threshold
            tp = int(np.count_nonzero(prediction & target))
            tn = int(np.count_nonzero(~prediction & ~target))
            fp = int(np.count_nonzero(prediction & ~target))
            fn = int(np.count_nonzero(~prediction & target))
            recall = tp / (tp + fn) if tp + fn else 0.0
            specificity = tn / (tn + fp) if tn + fp else 0.0
            precision = tp / (tp + fp) if tp + fp else 0.0
            balanced_accuracy = (recall + specificity) / 2
            accuracy = (tp + tn) / len(rows)
            f1 = (2 * precision * recall / (precision + recall)
                  if precision + recall else 0.0)
            candidate = {
                "direction": direction,
                "threshold": float(threshold),
                "balanced_accuracy": balanced_accuracy,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "specificity": specificity,
                "f1": f1,
                "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            }
            if best is None or (candidate["balanced_accuracy"], candidate["f1"]) > (
                    best["balanced_accuracy"], best["f1"]):
                best = candidate
    return best


def training_membership(csv_path):
    membership = set()
    if not csv_path.exists():
        return membership
    with csv_path.open("r", newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            if row.get("status") != "labeled":
                continue
            stem = Path(row.get("path", "")).stem
            match = re.match(r"^(pic_[0-9]+)_(.+)$", stem)
            if match:
                membership.add(f"{match.group(1)}/send_to_ground/{match.group(2)}.jpg")
    return membership


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("labels", nargs="?", default="send_to_ground_blur_labels.csv")
    parser.add_argument("--training-labels", default="digit_labels_send_to_ground_v5.csv")
    parser.add_argument("--output", default="send_to_ground_blur_analysis.json")
    args = parser.parse_args()

    label_path = Path(args.labels).resolve()
    with label_path.open("r", newline="", encoding="utf-8-sig") as handle:
        all_rows = list(csv.DictReader(handle))
    rows = [row for row in all_rows
            if row.get("status") == "labeled"
            and row.get("blur_label") in ("blurred", "clear")]

    datasets = {}
    dataset_thresholds = {}
    for dataset in sorted({row["dataset"] for row in all_rows}):
        group = [row for row in all_rows if row["dataset"] == dataset]
        labeled_group = [row for row in group
                         if row.get("status") == "labeled"
                         and row.get("blur_label") in ("blurred", "clear")]
        blurred = sum(row.get("blur_label") == "blurred" for row in group)
        clear = sum(row.get("blur_label") == "clear" for row in group)
        datasets[dataset] = {
            "total": len(group), "blurred": blurred, "clear": clear,
            "unlabeled": sum(row.get("status") == "unlabeled" for row in group),
            "blur_rate_labeled": blurred / (blurred + clear) if blurred + clear else None,
        }
        dataset_thresholds[dataset] = {
            metric: best_threshold(labeled_group, metric) for metric in METRICS
        }

    metric_summary = {}
    thresholds = {}
    y = np.asarray([row["blur_label"] == "blurred" for row in rows], dtype=np.float64)
    for metric in METRICS:
        metric_summary[metric] = {}
        for label in ("blurred", "clear"):
            metric_summary[metric][label] = quantiles(
                [float(row[metric]) for row in rows if row["blur_label"] == label])
        x = np.asarray([float(row[metric]) for row in rows], dtype=np.float64)
        metric_summary[metric]["point_biserial_correlation"] = float(
            np.corrcoef(x, y)[0, 1])
        thresholds[metric] = best_threshold(rows, metric)

    trained = training_membership(Path(args.training_labels).resolve())
    training_crosscheck = defaultdict(lambda: defaultdict(int))
    for row in all_rows:
        bucket = "used_for_digit_training" if row["source"] in trained else "not_used_for_digit_training"
        label = row.get("blur_label") or "unlabeled"
        training_crosscheck[bucket][label] += 1

    blurred_paths = [row["source"] for row in rows if row["blur_label"] == "blurred"]
    result = {
        "total_images": len(all_rows),
        "labeled_images": len(rows),
        "blurred_images": len(blurred_paths),
        "clear_images": sum(row["blur_label"] == "clear" for row in rows),
        "unlabeled_images": sum(row.get("status") == "unlabeled" for row in all_rows),
        "datasets": datasets,
        "dataset_thresholds": dataset_thresholds,
        "metric_summary": metric_summary,
        "best_single_metric_thresholds": thresholds,
        "digit_training_crosscheck": {
            key: dict(value) for key, value in training_crosscheck.items()},
        "blurred_paths": blurred_paths,
        "unlabeled_paths": [row["source"] for row in all_rows
                            if row.get("status") == "unlabeled"],
    }
    output = Path(args.output)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
