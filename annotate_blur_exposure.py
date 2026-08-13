"""Compare cropped plates with their lightweight binary images and label blur."""

from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass
from pathlib import Path
import tkinter as tk

import cv2
import numpy as np
from PIL import Image, ImageOps, ImageTk


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
CSV_FIELDS = (
    "source", "dataset", "filename", "blur_label", "status",
    "mean_gray", "laplacian_variance", "black_ratio", "otsu_threshold",
)


@dataclass
class Record:
    blur_label: str = ""
    status: str = "unlabeled"
    mean_gray: str = ""
    laplacian_variance: str = ""
    black_ratio: str = ""
    otsu_threshold: str = ""


def read_image(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"无法读取图片：{path}")
    return image


def lightweight_binary(image: np.ndarray) -> tuple[np.ndarray, float]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    threshold, binary = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary, float(threshold)


class BlurAnnotator:
    def __init__(self, root: tk.Tk, inputs: list[Path], csv_path: Path):
        self.root = root
        self.csv_path = csv_path.resolve()
        self.paths = self._collect(inputs)
        self.records = self._load_records()
        self.index = self._first_unlabeled()
        self.undo_stack: list[tuple[str, Record]] = []
        self.photo_refs: list[ImageTk.PhotoImage] = []

        root.title("send_to_ground 模糊与曝光标注")
        root.geometry("1200x760")
        root.minsize(900, 650)
        root.protocol("WM_DELETE_WINDOW", self.close)

        self.info = tk.Label(root, font=("Microsoft YaHei", 13), anchor="w",
                             justify="left")
        self.info.pack(fill="x", padx=16, pady=(10, 4))
        panel = tk.Frame(root, bg="#202020")
        panel.pack(fill="both", expand=True, padx=16, pady=6)
        left = tk.Frame(panel, bg="#202020")
        left.pack(side="left", fill="both", expand=True, padx=8, pady=8)
        right = tk.Frame(panel, bg="#202020")
        right.pack(side="right", fill="both", expand=True, padx=8, pady=8)
        tk.Label(left, text="原始 send_to_ground", fg="white", bg="#202020",
                 font=("Microsoft YaHei", 13, "bold")).pack()
        tk.Label(right, text="轻量二值图（3×3 + Otsu）", fg="white", bg="#202020",
                 font=("Microsoft YaHei", 13, "bold")).pack()
        self.original_label = tk.Label(left, bg="#202020")
        self.original_label.pack(fill="both", expand=True)
        self.binary_label = tk.Label(right, bg="#202020")
        self.binary_label.pack(fill="both", expand=True)
        self.state = tk.Label(root, font=("Microsoft YaHei", 20, "bold"),
                              fg="#0a67a3")
        self.state.pack(pady=3)
        help_text = (
            "B：模糊    C：清晰    S：不确定/跳过    U：撤销    "
            "←/→：浏览    Q/Esc：保存退出"
        )
        tk.Label(root, text=help_text, font=("Microsoft YaHei", 12)).pack(
            pady=(2, 10))

        root.bind("<Key>", self.on_key)
        root.bind("<Configure>", lambda _event: root.after_idle(self.render))
        if not self.paths:
            raise SystemExit("未找到 send_to_ground 图片")
        self.render()

    @staticmethod
    def _collect(inputs: list[Path]) -> list[Path]:
        paths: set[Path] = set()
        for item in inputs:
            item = item.resolve()
            if not item.is_dir():
                continue
            for path in item.rglob("*"):
                if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                    paths.add(path.resolve())
        return sorted(paths, key=lambda p: (
            p.parent.parent.name.lower(),
            int(p.stem) if p.stem.isdigit() else float("inf"),
            p.name.lower(),
        ))

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
                source = row.get("source", "")
                if source:
                    records[source] = Record(
                        blur_label=row.get("blur_label", ""),
                        status=row.get("status", "unlabeled"),
                        mean_gray=row.get("mean_gray", ""),
                        laplacian_variance=row.get("laplacian_variance", ""),
                        black_ratio=row.get("black_ratio", ""),
                        otsu_threshold=row.get("otsu_threshold", ""),
                    )
        return records

    def _first_unlabeled(self) -> int:
        for index, path in enumerate(self.paths):
            record = self.records.get(self._key(path), Record())
            if record.status == "unlabeled":
                return index
        return 0

    @staticmethod
    def _metrics(image: np.ndarray, binary: np.ndarray,
                 threshold: float) -> Record:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return Record(
            mean_gray=f"{float(gray.mean()):.4f}",
            laplacian_variance=f"{float(cv2.Laplacian(gray, cv2.CV_64F).var()):.4f}",
            black_ratio=f"{float(np.count_nonzero(binary == 0) / binary.size):.6f}",
            otsu_threshold=f"{threshold:.2f}",
        )

    def _save(self) -> None:
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.csv_path.with_suffix(self.csv_path.suffix + ".tmp")
        with temporary.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
            writer.writeheader()
            for path in self.paths:
                key = self._key(path)
                record = self.records.get(key, Record())
                writer.writerow({
                    "source": key,
                    "dataset": path.parent.parent.name,
                    "filename": path.name,
                    "blur_label": record.blur_label,
                    "status": record.status,
                    "mean_gray": record.mean_gray,
                    "laplacian_variance": record.laplacian_variance,
                    "black_ratio": record.black_ratio,
                    "otsu_threshold": record.otsu_threshold,
                })
        os.replace(temporary, self.csv_path)

    @staticmethod
    def _fit(image: Image.Image, max_width: int, max_height: int,
             binary: bool = False) -> Image.Image:
        resized = ImageOps.contain(
            image, (max(120, max_width), max(120, max_height)),
            method=(Image.Resampling.NEAREST if binary else Image.Resampling.LANCZOS))
        return resized

    def render(self) -> None:
        if not self.root.winfo_exists():
            return
        path = self.paths[self.index]
        key = self._key(path)
        record = self.records.get(key, Record())
        image = read_image(path)
        binary, threshold = lightweight_binary(image)
        metrics = self._metrics(image, binary, threshold)
        done = sum(r.status == "labeled" for r in self.records.values())
        blurred = sum(r.blur_label == "blurred" for r in self.records.values())
        clear = sum(r.blur_label == "clear" for r in self.records.values())
        self.info.config(text=(
            f"{self.index + 1}/{len(self.paths)}   已标 {done}   模糊 {blurred}   清晰 {clear}\n"
            f"{key}\n平均灰度 {metrics.mean_gray}   Laplacian {metrics.laplacian_variance}   "
            f"黑像素比 {metrics.black_ratio}   Otsu阈值 {metrics.otsu_threshold}"
        ))
        names = {"blurred": "模糊", "clear": "清晰", "uncertain": "不确定", "": "未标"}
        self.state.config(text=f"当前标注：{names.get(record.blur_label, record.blur_label)}")

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_pil = Image.fromarray(rgb)
        binary_pil = Image.fromarray(binary)
        max_width = max(320, self.root.winfo_width() // 2 - 60)
        max_height = max(320, self.root.winfo_height() - 230)
        original_show = self._fit(original_pil, max_width, max_height)
        binary_show = self._fit(binary_pil, max_width, max_height, binary=True)
        self.photo_refs = [
            ImageTk.PhotoImage(original_show), ImageTk.PhotoImage(binary_show)]
        self.original_label.config(image=self.photo_refs[0])
        self.binary_label.config(image=self.photo_refs[1])

    def _record(self, label: str, status: str) -> None:
        path = self.paths[self.index]
        key = self._key(path)
        old = self.records.get(key, Record())
        self.undo_stack.append((key, Record(**old.__dict__)))
        image = read_image(path)
        binary, threshold = lightweight_binary(image)
        record = self._metrics(image, binary, threshold)
        record.blur_label = label
        record.status = status
        self.records[key] = record
        self._save()
        self._advance()

    def _advance(self) -> None:
        for offset in range(1, len(self.paths) + 1):
            candidate = (self.index + offset) % len(self.paths)
            record = self.records.get(self._key(self.paths[candidate]), Record())
            if record.status == "unlabeled":
                self.index = candidate
                self.render()
                return
        self.render()

    def undo(self) -> None:
        if not self.undo_stack:
            return
        key, old = self.undo_stack.pop()
        self.records[key] = old
        for index, path in enumerate(self.paths):
            if self._key(path) == key:
                self.index = index
                break
        self._save()
        self.render()

    def on_key(self, event: tk.Event) -> None:
        key = event.keysym.lower()
        if key == "b":
            self._record("blurred", "labeled")
            return
        if key == "c":
            self._record("clear", "labeled")
            return
        if key == "s":
            self._record("uncertain", "skipped")
            return
        if key == "u" or (key == "z" and event.state & 0x4):
            self.undo()
            return
        if key == "left":
            self.index = max(0, self.index - 1)
        elif key == "right":
            self.index = min(len(self.paths) - 1, self.index + 1)
        elif key in {"q", "escape"}:
            self.close()
            return
        self.render()

    def close(self) -> None:
        self._save()
        self.root.destroy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", help="send_to_ground directories")
    parser.add_argument("--output", default="send_to_ground_blur_labels.csv")
    args = parser.parse_args()
    root = tk.Tk()
    BlurAnnotator(root, [Path(item) for item in args.inputs], Path(args.output))
    root.mainloop()


if __name__ == "__main__":
    main()
