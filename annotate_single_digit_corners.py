"""Manually rectify and label single-digit plates with four mouse clicks.

Click corners in the digit's upright orientation:
    top-left -> top-right -> bottom-right -> bottom-left

The tool saves a color 100x100 plate, an Otsu-binary 100x100 plate, and a
32x32 single-digit CNN input. Progress is recorded in a CSV and can be resumed.
"""

import argparse
import csv
import hashlib
import os
import re
from pathlib import Path

import cv2
import numpy as np


SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
CSV_FIELDS = ("source", "label", "points", "color_path", "binary_path",
              "digit_path", "split", "status")


def imread_unicode(path):
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite_unicode(path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix or ".png"
    ok, encoded = cv2.imencode(suffix, image)
    if not ok:
        raise RuntimeError("Failed to encode {}".format(path))
    encoded.tofile(str(path))


def collect_images(inputs):
    paths = set()
    for item in inputs:
        path = Path(item).resolve()
        if path.is_dir():
            for candidate in path.rglob("*"):
                if candidate.is_file() and candidate.suffix.lower() in SUFFIXES:
                    paths.add(candidate.resolve())
        elif path.is_file() and path.suffix.lower() in SUFFIXES:
            paths.add(path)
    return sorted(paths, key=lambda p: str(p).lower())


def read_records(csv_path):
    records = {}
    if not csv_path.exists():
        return records
    with csv_path.open("r", newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            source = row.get("source", "")
            if source:
                records[source] = row
    return records


def write_records(csv_path, records):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = csv_path.with_suffix(csv_path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for source in sorted(records):
            writer.writerow(records[source])
    os.replace(str(temporary), str(csv_path))


def rectify(image, points, width, height):
    source = np.asarray(points, dtype=np.float32)
    destination = np.asarray([
        [0, 0], [width - 1, 0],
        [width - 1, height - 1], [0, height - 1]
    ], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(source, destination)
    return cv2.warpPerspective(
        image, matrix, (width, height), flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))


def make_binary(color):
    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def make_digit(binary, margin_ratio):
    height, width = binary.shape[:2]
    margin_x = max(0, min(width // 4, int(round(width * margin_ratio))))
    margin_y = max(0, min(height // 4, int(round(height * margin_ratio))))
    cropped = binary[margin_y:height - margin_y,
                     margin_x:width - margin_x]
    crop_h, crop_w = cropped.shape[:2]
    side = max(crop_h, crop_w)
    canvas = np.full((side, side), 255, dtype=np.uint8)
    y = (side - crop_h) // 2
    x = (side - crop_w) // 2
    canvas[y:y + crop_h, x:x + crop_w] = cropped
    return cv2.resize(canvas, (32, 32), interpolation=cv2.INTER_AREA)


class CornerAnnotator:
    def __init__(self, paths, output, csv_path, plate_width, plate_height,
                 margin_ratio, max_width, max_height, split,
                 label_from_parent):
        self.paths = paths
        self.output = output.resolve()
        self.csv_path = csv_path.resolve()
        self.plate_width = plate_width
        self.plate_height = plate_height
        self.margin_ratio = margin_ratio
        self.max_width = max_width
        self.max_height = max_height
        self.split = split
        self.label_from_parent = label_from_parent
        self.records = read_records(self.csv_path)
        self.index = self._first_unfinished()
        self.points = []
        self.label = None
        self.image = None
        self.display = None
        self.scale = 1.0
        self.window = "Manual single-digit corners"

    def _first_unfinished(self):
        for index, path in enumerate(self.paths):
            record = self.records.get(str(path), {})
            if record.get("status") not in ("saved", "skipped"):
                return index
        return len(self.paths)

    def _load(self):
        self.points = []
        self.label = self._label_for_path(self.paths[self.index])
        self.image = imread_unicode(self.paths[self.index])
        if self.image is None:
            raise RuntimeError("Cannot read {}".format(self.paths[self.index]))
        height, width = self.image.shape[:2]
        self.scale = min(1.0, self.max_width / float(width),
                         self.max_height / float(height))
        self.display = cv2.resize(
            self.image, None, fx=self.scale, fy=self.scale,
            interpolation=cv2.INTER_AREA)

    def _label_for_path(self, path):
        if not self.label_from_parent:
            return None
        match = re.match(r"^([0-9])(?:_|$)", path.parent.name)
        return match.group(1) if match else None

    def _render(self):
        canvas = self.display.copy()
        shown = [(int(round(x * self.scale)), int(round(y * self.scale)))
                 for x, y in self.points]
        names = ("TL", "TR", "BR", "BL")
        for index, point in enumerate(shown):
            cv2.circle(canvas, point, 7, (0, 255, 0), -1)
            cv2.putText(canvas, names[index], (point[0] + 8, point[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
        for index in range(1, len(shown)):
            cv2.line(canvas, shown[index - 1], shown[index], (0, 255, 0), 2)
        if len(shown) == 4:
            cv2.line(canvas, shown[3], shown[0], (0, 255, 0), 2)
        text = "[{}/{}] label={} points={}/4  ENTER=save  0-9=label  U=undo  R=reset  S=skip  Q=quit".format(
            self.index + 1, len(self.paths),
            "-" if self.label is None else self.label, len(self.points))
        cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 35), (30, 30, 30), -1)
        cv2.putText(canvas, text, (8, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow(self.window, canvas)

        if len(self.points) == 4:
            color = rectify(self.image, self.points,
                            self.plate_width, self.plate_height)
            binary = make_binary(color)
            digit = make_digit(binary, self.margin_ratio)
            preview_height = self.plate_height
            preview = np.hstack([
                color,
                cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR),
                cv2.resize(cv2.cvtColor(digit, cv2.COLOR_GRAY2BGR),
                           (preview_height, preview_height),
                           interpolation=cv2.INTER_NEAREST)
            ])
            cv2.imshow("Rectified: color | binary | CNN 32x32", preview)
        else:
            try:
                cv2.destroyWindow("Rectified: color | binary | CNN 32x32")
            except cv2.error:
                pass

    def _mouse(self, event, x, y, _flags, _userdata):
        if event == cv2.EVENT_LBUTTONDOWN and len(self.points) < 4:
            self.points.append((x / self.scale, y / self.scale))

    def _save(self):
        if len(self.points) != 4 or self.label is None:
            print("Need four points and a digit label before saving")
            return False
        source = self.paths[self.index]
        color = rectify(self.image, self.points,
                        self.plate_width, self.plate_height)
        binary = make_binary(color)
        digit = make_digit(binary, self.margin_ratio)
        token = hashlib.sha1(str(source).encode("utf-8")).hexdigest()[:8]
        stem = "{}_{}".format(source.stem, token)
        color_path = self.output / "plate_color" / self.label / (stem + ".png")
        binary_path = self.output / "plate_binary" / self.label / (stem + ".png")
        digit_path = self.output / "digits_32" / self.label / (stem + ".png")
        imwrite_unicode(color_path, color)
        imwrite_unicode(binary_path, binary)
        imwrite_unicode(digit_path, digit)
        self.records[str(source)] = {
            "source": str(source),
            "label": self.label,
            "points": ";".join("{:.2f},{:.2f}".format(x, y)
                               for x, y in self.points),
            "color_path": str(color_path),
            "binary_path": str(binary_path),
            "digit_path": str(digit_path),
            "split": self.split,
            "status": "saved",
        }
        write_records(self.csv_path, self.records)
        print("SAVED label={} {}".format(self.label, source.name))
        return True

    def _skip(self):
        source = self.paths[self.index]
        self.records[str(source)] = {
            "source": str(source), "label": "", "points": "",
            "color_path": "", "binary_path": "", "digit_path": "",
            "split": self.split, "status": "skipped",
        }
        write_records(self.csv_path, self.records)
        print("SKIPPED {}".format(source.name))

    def _advance(self):
        if self.index + 1 >= len(self.paths):
            print("All images finished")
            return False
        self.index += 1
        self._load()
        return True

    def run(self):
        if not self.paths:
            raise SystemExit("No supported images found")
        if self.index >= len(self.paths):
            print("All images were already finished")
            return
        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window, self._mouse)
        self._load()
        while True:
            self._render()
            key = cv2.waitKey(20) & 0xFF
            if ord("0") <= key <= ord("9"):
                self.label = chr(key)
            elif key in (13, 10):
                if self._save() and not self._advance():
                    break
            elif key in (ord("u"), ord("U"), 8):
                if self.points:
                    self.points.pop()
            elif key in (ord("r"), ord("R")):
                self.points = []
            elif key in (ord("s"), ord("S")):
                self._skip()
                if not self._advance():
                    break
            elif key in (ord("q"), ord("Q"), 27):
                break
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", help="image files or directories")
    parser.add_argument("--output", default="single_digit_dataset")
    parser.add_argument("--csv", default="single_digit_corners.csv")
    parser.add_argument("--plate-width", type=int, default=50,
                        help="rectified single plate width")
    parser.add_argument("--plate-height", type=int, default=100,
                        help="rectified single plate height")
    parser.add_argument("--inner-margin", type=float, default=0.03)
    parser.add_argument("--split", choices=("train", "val", "test"),
                        default="train",
                        help="dataset split assigned to every saved image")
    parser.add_argument("--label-from-parent", action="store_true",
                        help="infer label from parent names such as 3_above")
    parser.add_argument("--max-width", type=int, default=1400)
    parser.add_argument("--max-height", type=int, default=850)
    args = parser.parse_args()
    paths = collect_images(args.inputs)
    CornerAnnotator(
        paths, Path(args.output), Path(args.csv), args.plate_width,
        args.plate_height, args.inner_margin, args.max_width,
        args.max_height, args.split, args.label_from_parent).run()


if __name__ == "__main__":
    main()
