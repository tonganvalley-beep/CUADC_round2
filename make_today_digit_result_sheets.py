"""Create compact original/binary/prediction sheets for today's plate crops."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import cv2
import numpy as np


def read_image(path: Path, mode: int) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), mode)
    if image is None:
        raise RuntimeError(f"cannot read {path}")
    return image


def write_image(path: Path, image: np.ndarray) -> None:
    ok, encoded = cv2.imencode(path.suffix, image)
    if not ok:
        raise RuntimeError(f"cannot encode {path}")
    encoded.tofile(path)


def make_card(row: dict[str, str], csv_dir: Path) -> tuple[str, np.ndarray]:
    source_path = (csv_dir / row["path"]).resolve()
    binary_path = (csv_dir / row["binary_path"]).resolve()
    original = read_image(source_path, cv2.IMREAD_COLOR)
    binary = read_image(binary_path, cv2.IMREAD_GRAYSCALE)
    original = cv2.resize(original, (150, 150), interpolation=cv2.INTER_CUBIC)
    binary = cv2.resize(binary, (150, 150), interpolation=cv2.INTER_NEAREST)
    binary = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

    card = np.full((198, 314, 3), 246, dtype=np.uint8)
    card[44:194, 4:154] = original
    card[44:194, 160:310] = binary
    dataset = source_path.parent.parent.name
    title = f"{dataset}/{source_path.name}  CNN={row['predicted']}  {row['confidence_status']}"
    confidence = f"L={float(row['left_confidence']):.3f}  R={float(row['right_confidence']):.3f}"
    cv2.putText(card, title, (5, 17), cv2.FONT_HERSHEY_SIMPLEX,
                0.43, (25, 25, 25), 1, cv2.LINE_AA)
    cv2.putText(card, confidence, (5, 37), cv2.FONT_HERSHEY_SIMPLEX,
                0.43, (25, 25, 25), 1, cv2.LINE_AA)
    cv2.rectangle(card, (3, 43), (154, 194), (115, 115, 115), 1)
    cv2.rectangle(card, (159, 43), (310, 194), (115, 115, 115), 1)
    return dataset, card


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--review", type=Path,
        default=Path("digit_review_pic22_23_v2_shadow_dynamic/review.csv"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("digit_review_pic22_23_v2_shadow_dynamic"),
    )
    args = parser.parse_args()
    review_path = args.review.resolve()
    with review_path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))

    groups: dict[str, list[np.ndarray]] = {}
    for row in rows:
        dataset, card = make_card(row, review_path.parent)
        groups.setdefault(dataset, []).append(card)

    args.output.mkdir(parents=True, exist_ok=True)
    for dataset, cards in groups.items():
        columns = 4
        rows_count = math.ceil(len(cards) / columns)
        blank = np.full_like(cards[0], 246)
        cards = cards + [blank] * (rows_count * columns - len(cards))
        sheet_rows = [cv2.hconcat(cards[index:index + columns])
                      for index in range(0, len(cards), columns)]
        sheet = cv2.vconcat(sheet_rows)
        path = args.output / f"{dataset}_today_binary_results.jpg"
        write_image(path, sheet)
        print(path.resolve())


if __name__ == "__main__":
    main()
