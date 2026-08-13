"""Generate an auditable exposure report for saved frames and plate crops.

Each contact-sheet tile contains the color input, the grayscale image used by
the controller, and its Otsu binary result.  The CSV keeps all numeric metrics
so a human label can be compared with the controller decision.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np

try:
    from exposure_controller import evaluate_exposure, prepare_exposure_views
except ImportError:
    # 兼容当前工作区中尚未恢复正式文件名的调试版本。
    from exposure_controller_M import evaluate_exposure

    def prepare_exposure_views(image, source):
        margin = 0.06 if source == "plate" else 0.15
        height, width = image.shape[:2]
        mx = min(max(int(width * margin), 0), max(0, width // 2 - 1))
        my = min(max(int(height * margin), 0), max(0, height // 2 - 1))
        roi = image[my : height - my, mx : width - mx]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        threshold, binary = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        return roi, gray, binary, float(threshold)


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def read_image(path: Path) -> Optional[np.ndarray]:
    """Read an image, including paths containing non-ASCII characters."""

    try:
        encoded = np.fromfile(str(path), dtype=np.uint8)
        return cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    except (OSError, ValueError, cv2.error):
        return None


def write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix or ".jpg"
    ok, encoded = cv2.imencode(suffix, image)
    if not ok:
        raise OSError("failed to encode {}".format(path))
    encoded.tofile(str(path))


def iter_images(path: Path) -> Iterable[Path]:
    if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
        yield path
    elif path.is_dir():
        yield from sorted(
            (item for item in path.iterdir() if item.suffix.lower() in IMAGE_SUFFIXES),
            key=lambda item: (
                int(item.stem.split("_")[0])
                if item.stem.split("_")[0].isdigit()
                else 10**12,
                item.name,
            ),
        )


def load_labels(path: Optional[Path]) -> Dict[str, Tuple[str, str]]:
    if path is None:
        return {}
    labels: Dict[str, Tuple[str, str]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            key = (row.get("path") or "").replace("\\", "/")
            label = (row.get("human_label") or "").strip().upper()
            if key and label:
                labels[key] = (label, (row.get("human_note") or "").strip())
    return labels


def load_label_ranges(path: Optional[Path]) -> List[Tuple[str, int, int, str, str]]:
    if path is None:
        return []
    ranges = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            ranges.append(
                (
                    (row.get("dataset") or "").replace("\\", "/").rstrip("/"),
                    int(row["start"]),
                    int(row["end"]),
                    row["human_label"].strip().upper(),
                    (row.get("human_note") or "").strip(),
                )
            )
    return ranges


def find_label(
    relative: str,
    stem: str,
    labels: Dict[str, Tuple[str, str]],
    ranges: List[Tuple[str, int, int, str, str]],
) -> Tuple[str, str]:
    if relative in labels:
        return labels[relative]
    prefix = stem.split("_")[0]
    if not prefix.isdigit():
        return "", ""
    number = int(prefix)
    parent = str(Path(relative).parent).replace("\\", "/")
    for dataset, start, end, label, note in ranges:
        if parent == dataset and start <= number <= end:
            return label, note
    return "", ""


def _fit(image: np.ndarray, width: int, height: int, gray: bool = False) -> np.ndarray:
    if gray and image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    h, w = image.shape[:2]
    scale = min(width / max(w, 1), height / max(h, 1))
    resized = cv2.resize(
        image,
        (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
        interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_NEAREST,
    )
    canvas = np.full((height, width, 3), 245, dtype=np.uint8)
    y = (height - resized.shape[0]) // 2
    x = (width - resized.shape[1]) // 2
    canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return canvas


def make_tile(
    image: np.ndarray,
    gray: np.ndarray,
    binary: np.ndarray,
    name: str,
    state: str,
    human_label: str,
    p50: float,
    bright_ratio: float,
    foreground_ratio: float,
    otsu_threshold: float,
    blind: bool,
) -> np.ndarray:
    tile = np.full((154, 354, 3), 255, dtype=np.uint8)
    for index, view in enumerate((image, gray, binary)):
        tile[4:104, 4 + index * 116 : 114 + index * 116] = _fit(
            view, 110, 100, gray=index > 0
        )
        cv2.putText(
            tile,
            ("COLOR", "GRAY", "OTSU")[index],
            (7 + index * 116, 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.32,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )
    mismatch = human_label and human_label != state
    color = (
        (35, 35, 35)
        if blind
        else (
            (0, 0, 220)
            if mismatch
            else ((0, 130, 0) if state == "NORMAL" else (0, 90, 220))
        )
    )
    cv2.putText(tile, name, (4, 119), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (20, 20, 20), 1, cv2.LINE_AA)
    status = "USER: ____" if blind else "P:{} H:{}".format(
        state[:4], human_label[:4] or "?"
    )
    cv2.putText(tile, status, (88, 119), cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)
    if not blind:
        cv2.putText(
            tile,
            "p50={:.0f} clip={:.0%} ink={:.0%} otsu={:.0f}".format(
                p50, bright_ratio, foreground_ratio, otsu_threshold
            ),
            (4, 140),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (30, 30, 30),
            1,
            cv2.LINE_AA,
        )
    return tile


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=Path("exposure_analysis"))
    parser.add_argument("--labels", type=Path, default=None)
    parser.add_argument("--label-ranges", type=Path, default=None)
    parser.add_argument("--source", choices=("auto", "plate", "frame"), default="auto")
    parser.add_argument("--columns", type=int, default=3)
    parser.add_argument("--rows", type=int, default=6)
    parser.add_argument(
        "--blind",
        action="store_true",
        help="hide controller and human labels in contact sheets",
    )
    args = parser.parse_args()

    labels = load_labels(args.labels)
    label_ranges = load_label_ranges(args.label_ranges)
    args.output.mkdir(parents=True, exist_ok=True)
    fields = [
        "path", "source", "human_label", "human_note", "program_state", "match", "reason",
        "p05", "p50", "p90", "p95", "p99", "bright_ratio", "white_ratio",
        "channel_clip_ratio", "multi_channel_clip_ratio", "highlight_tile_fraction",
        "max_tile_highlight_ratio", "dark_ratio", "contrast", "edge_p90",
        "foreground_ratio", "otsu_threshold",
    ]
    all_rows = []
    label_template_rows = []

    for input_path in args.inputs:
        tiles = []
        dataset = input_path.stem if input_path.is_file() else input_path.name
        parent = input_path.parent.name
        if parent.startswith("pic_"):
            dataset = "{}_{}".format(parent, dataset)
        for path in iter_images(input_path):
            image = read_image(path)
            if image is None:
                continue
            source = args.source
            if source == "auto":
                source = "frame" if max(image.shape[:2]) > 500 else "plate"
            result = evaluate_exposure(image, source)
            _, gray, binary, otsu_threshold = prepare_exposure_views(image, source)
            relative = path.as_posix()
            human_label, human_note = find_label(
                relative, path.stem, labels, label_ranges
            )
            row = {
                "path": relative,
                "source": source,
                "human_label": human_label,
                "human_note": human_note,
                "program_state": result.state,
                "match": (
                    "" if human_label in ("", "INVALID")
                    else str(human_label == result.state)
                ),
                "reason": result.reason,
            }
            row.update(result.metrics)
            all_rows.append(row)
            label_template_rows.append(
                {"path": relative, "user_label": "", "user_note": ""}
            )
            tiles.append(
                make_tile(
                    image, gray, binary, path.name, result.state, human_label,
                    result.metrics["p50"], result.metrics["bright_ratio"],
                    result.metrics["foreground_ratio"], otsu_threshold,
                    args.blind,
                )
            )

        page_size = max(1, args.columns * args.rows)
        for page_index in range(0, len(tiles), page_size):
            page_tiles = tiles[page_index : page_index + page_size]
            sheet = np.full(
                (args.rows * 154, args.columns * 354, 3), 232, dtype=np.uint8
            )
            for index, tile in enumerate(page_tiles):
                y = (index // args.columns) * 154
                x = (index % args.columns) * 354
                sheet[y : y + 154, x : x + 354] = tile
            write_image(
                args.output / "{}_page_{:02d}.jpg".format(dataset, page_index // page_size + 1),
                sheet,
            )

    if not args.blind:
        with (args.output / "exposure_comparison.csv").open(
            "w", encoding="utf-8-sig", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(all_rows)
    with (args.output / "user_label_template.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("path", "user_label", "user_note")
        )
        writer.writeheader()
        writer.writerows(label_template_rows)
    print("wrote {} images to {}".format(len(all_rows), args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
