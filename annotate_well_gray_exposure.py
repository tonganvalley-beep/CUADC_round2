"""Label overexposure from the original crop and classifier grayscale input.

This annotator is intentionally specific to ``detect_well_exposure(3).py``:
the right panel is exactly the BGR-to-grayscale image seen by its classifier.
No Otsu result or color-channel metric is shown or used.
"""

from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
CSV_FIELDS = ("source", "dataset", "filename", "exposure_label", "status")
WINDOW_NAME = "detect_well_exposure(3) grayscale exposure labeling"
CANVAS_WIDTH = 1360
CANVAS_HEIGHT = 800
HEADER_HEIGHT = 105
FOOTER_HEIGHT = 105
PANEL_GAP = 20


@dataclass
class Record:
    exposure_label: str = ""
    status: str = "unlabeled"


def read_image(path: Path) -> np.ndarray:
    """Read an image from a path that may contain non-ASCII characters."""

    encoded = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError("cannot read image: {}".format(path))
    return image


def classifier_gray(image: np.ndarray) -> np.ndarray:
    """Return the grayscale pixels used by detect_well_exposure(3).py."""

    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def collect_images(inputs: Iterable[Path]) -> list[Path]:
    paths: set[Path] = set()
    for item in inputs:
        item = item.resolve()
        if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES:
            paths.add(item)
        elif item.is_dir():
            for path in item.rglob("*"):
                if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                    paths.add(path.resolve())

    def order(path: Path) -> tuple[str, int, str]:
        number = int(path.stem) if path.stem.isdigit() else 10**12
        dataset = path.parent.parent.name.lower()
        return dataset, number, path.name.lower()

    return sorted(paths, key=order)


class GrayExposureAnnotator:
    def __init__(self, paths: list[Path], csv_path: Path) -> None:
        if not paths:
            raise SystemExit("no images found")
        self.paths = paths
        self.csv_path = csv_path.resolve()
        self.records = self._load_records()
        self.index = self._first_unlabeled()
        self.undo_stack: list[tuple[str, Record]] = []
        self.running = True
        self.buttons: list[tuple[tuple[int, int, int, int], str]] = []

    def _key(self, path: Path) -> str:
        try:
            return path.relative_to(self.csv_path.parent).as_posix()
        except ValueError:
            return str(path)

    def _load_records(self) -> dict[str, Record]:
        records: dict[str, Record] = {}
        if not self.csv_path.exists():
            return records
        with self.csv_path.open("r", newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                source = (row.get("source") or "").strip()
                if source:
                    records[source] = Record(
                        exposure_label=(row.get("exposure_label") or "").strip(),
                        status=(row.get("status") or "unlabeled").strip(),
                    )
        return records

    def _first_unlabeled(self) -> int:
        for index, path in enumerate(self.paths):
            if self.records.get(self._key(path), Record()).status == "unlabeled":
                return index
        return 0

    def save(self) -> None:
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.csv_path.with_suffix(self.csv_path.suffix + ".tmp")
        with temporary.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
            writer.writeheader()
            for path in self.paths:
                key = self._key(path)
                record = self.records.get(key, Record())
                writer.writerow(
                    {
                        "source": key,
                        "dataset": path.parent.parent.name,
                        "filename": path.name,
                        "exposure_label": record.exposure_label,
                        "status": record.status,
                    }
                )
        os.replace(temporary, self.csv_path)

    @staticmethod
    def _fit(image: np.ndarray, width: int, height: int) -> np.ndarray:
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        source_height, source_width = image.shape[:2]
        scale = min(width / max(source_width, 1), height / max(source_height, 1))
        target = (
            max(1, int(round(source_width * scale))),
            max(1, int(round(source_height * scale))),
        )
        # Nearest-neighbour display preserves the original gray values.
        resized = cv2.resize(image, target, interpolation=cv2.INTER_NEAREST)
        panel = np.full((height, width, 3), 30, dtype=np.uint8)
        y = (height - resized.shape[0]) // 2
        x = (width - resized.shape[1]) // 2
        panel[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
        return panel

    @staticmethod
    def _text(
        canvas: np.ndarray,
        value: str,
        origin: tuple[int, int],
        scale: float = 0.62,
        color: tuple[int, int, int] = (235, 235, 235),
        thickness: int = 1,
    ) -> None:
        cv2.putText(
            canvas,
            value,
            origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            thickness,
            cv2.LINE_AA,
        )

    def _button(
        self,
        canvas: np.ndarray,
        x1: int,
        x2: int,
        label: str,
        action: str,
        color: tuple[int, int, int],
    ) -> None:
        y1 = CANVAS_HEIGHT - 75
        y2 = CANVAS_HEIGHT - 20
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, -1)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (230, 230, 230), 1)
        size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        tx = x1 + max(4, (x2 - x1 - size[0]) // 2)
        ty = y1 + (y2 - y1 + size[1]) // 2
        self._text(canvas, label, (tx, ty), 0.55, (255, 255, 255), 1)
        self.buttons.append(((x1, y1, x2, y2), action))

    def render(self) -> np.ndarray:
        path = self.paths[self.index]
        key = self._key(path)
        record = self.records.get(key, Record())
        image = read_image(path)
        gray = classifier_gray(image)

        labeled = sum(r.status == "labeled" for r in self.records.values())
        over = sum(r.exposure_label == "OVEREXPOSED" for r in self.records.values())
        normal = sum(r.exposure_label == "NORMAL" for r in self.records.values())
        uncertain = sum(r.exposure_label == "UNCERTAIN" for r in self.records.values())

        canvas = np.full((CANVAS_HEIGHT, CANVAS_WIDTH, 3), 20, dtype=np.uint8)
        self._text(
            canvas,
            "{}/{}   labeled {}   OVER {}   NORMAL {}   uncertain {}".format(
                self.index + 1, len(self.paths), labeled, over, normal, uncertain
            ),
            (20, 30),
            0.72,
            (245, 245, 245),
            2,
        )
        short_name = "{}/{}".format(path.parent.parent.name, path.name)
        self._text(canvas, short_name, (20, 63), 0.66, (210, 210, 210), 1)
        state_color = {
            "OVEREXPOSED": (70, 90, 255),
            "NORMAL": (80, 220, 100),
            "UNCERTAIN": (80, 210, 230),
        }.get(record.exposure_label, (190, 190, 190))
        self._text(
            canvas,
            "current: {}".format(record.exposure_label or "UNLABELED"),
            (20, 92),
            0.62,
            state_color,
            2,
        )

        panel_width = (CANVAS_WIDTH - 3 * PANEL_GAP) // 2
        panel_height = CANVAS_HEIGHT - HEADER_HEIGHT - FOOTER_HEIGHT
        left_x = PANEL_GAP
        right_x = 2 * PANEL_GAP + panel_width
        top = HEADER_HEIGHT
        canvas[top : top + panel_height, left_x : left_x + panel_width] = self._fit(
            image, panel_width, panel_height
        )
        canvas[top : top + panel_height, right_x : right_x + panel_width] = self._fit(
            gray, panel_width, panel_height
        )
        self._text(canvas, "ORIGINAL", (left_x + 12, top + 28), 0.65, (60, 230, 255), 2)
        self._text(
            canvas,
            "GRAYSCALE - CLASSIFIER INPUT",
            (right_x + 12, top + 28),
            0.65,
            (60, 230, 255),
            2,
        )

        self.buttons = []
        self._button(canvas, 20, 130, "< PREV", "prev", (75, 75, 75))
        self._button(canvas, 155, 350, "O / 1  OVER", "over", (40, 55, 205))
        self._button(canvas, 375, 570, "N / 0  NORMAL", "normal", (40, 145, 55))
        self._button(canvas, 595, 770, "S  UNCERTAIN", "uncertain", (45, 135, 155))
        self._button(canvas, 795, 900, "U  UNDO", "undo", (105, 75, 35))
        self._button(canvas, 925, 1035, "NEXT >", "next", (75, 75, 75))
        self._button(canvas, 1170, 1335, "Q / ESC  EXIT", "exit", (80, 55, 80))
        return canvas

    def _advance_to_unlabeled(self) -> None:
        for offset in range(1, len(self.paths) + 1):
            candidate = (self.index + offset) % len(self.paths)
            record = self.records.get(self._key(self.paths[candidate]), Record())
            if record.status == "unlabeled":
                self.index = candidate
                return

    def record(self, label: str, status: str = "labeled") -> None:
        key = self._key(self.paths[self.index])
        previous = self.records.get(key, Record())
        self.undo_stack.append((key, Record(previous.exposure_label, previous.status)))
        self.records[key] = Record(label, status)
        self.save()
        self._advance_to_unlabeled()

    def undo(self) -> None:
        if not self.undo_stack:
            return
        key, previous = self.undo_stack.pop()
        self.records[key] = previous
        for index, path in enumerate(self.paths):
            if self._key(path) == key:
                self.index = index
                break
        self.save()

    def action(self, action: str) -> None:
        if action == "over":
            self.record("OVEREXPOSED")
        elif action == "normal":
            self.record("NORMAL")
        elif action == "uncertain":
            self.record("UNCERTAIN", "skipped")
        elif action == "undo":
            self.undo()
        elif action == "prev":
            self.index = max(0, self.index - 1)
        elif action == "next":
            self.index = min(len(self.paths) - 1, self.index + 1)
        elif action == "exit":
            self.running = False

    def on_mouse(self, event: int, x: int, y: int, _flags: int, _data: object) -> None:
        if event != cv2.EVENT_LBUTTONUP:
            return
        for (x1, y1, x2, y2), action in self.buttons:
            if x1 <= x <= x2 and y1 <= y <= y2:
                self.action(action)
                break

    def run(self) -> None:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, CANVAS_WIDTH, CANVAS_HEIGHT)
        cv2.setMouseCallback(WINDOW_NAME, self.on_mouse)
        try:
            while self.running:
                cv2.imshow(WINDOW_NAME, self.render())
                raw_key = cv2.waitKeyEx(50)
                if raw_key != -1:
                    key = raw_key & 0xFF
                    if key in (ord("o"), ord("O"), ord("1")):
                        self.action("over")
                    elif key in (ord("n"), ord("N"), ord("0")):
                        self.action("normal")
                    elif key in (ord("s"), ord("S")):
                        self.action("uncertain")
                    elif key in (ord("u"), ord("U"), 8):
                        self.action("undo")
                    elif key in (ord("a"), ord("A")) or raw_key in (2424832, 65361):
                        self.action("prev")
                    elif key in (ord("d"), ord("D")) or raw_key in (2555904, 65363):
                        self.action("next")
                    elif key in (ord("q"), ord("Q"), 27):
                        self.action("exit")
                try:
                    if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                        self.running = False
                except cv2.error:
                    self.running = False
        finally:
            self.save()
            cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("well_exposure_gray_labels.csv"),
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="validate inputs and print the image count without opening a window",
    )
    args = parser.parse_args()
    paths = collect_images(args.inputs)
    if args.check:
        print("found {} images".format(len(paths)))
        for directory in args.inputs:
            count = sum(1 for path in paths if directory.resolve() in path.parents)
            print("{}: {}".format(directory, count))
        return
    GrayExposureAnnotator(paths, args.output).run()


if __name__ == "__main__":
    main()
