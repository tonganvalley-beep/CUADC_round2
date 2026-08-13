"""Review rectified two-digit candidates and save resumable CSV labels.

Only rows with status ``labeled`` are consumed by train_digit_cnn.py.
Pattern-only and empty/false-positive detections are retained in the CSV for
auditing but are excluded from digit classification training.
"""

from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass
from pathlib import Path
import tkinter as tk
from tkinter import messagebox

from PIL import Image, ImageOps, ImageTk


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
CSV_FIELDS = ("path", "label", "status", "split_gap")


@dataclass
class Record:
    label: str = ""
    status: str = "unlabeled"
    split_gap: int = 3


class CandidateAnnotator:
    def __init__(self, root: tk.Tk, image_dir: Path, csv_path: Path, gap: int):
        self.root = root
        self.csv_path = csv_path.resolve()
        self.gap = max(0, gap)
        self.paths = sorted(
            (p.resolve() for p in image_dir.resolve().rglob("*")
             if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES),
            key=lambda p: str(p).lower(),
        )
        self.records = self._load_records()
        self.index = self._first_unlabeled()
        self.typed = ""
        self.undo_stack: list[tuple[str, Record]] = []
        self.photo_refs: list[ImageTk.PhotoImage] = []

        root.title("CUADC 数字牌候选标注")
        root.geometry("1100x780")
        root.minsize(850, 640)
        root.protocol("WM_DELETE_WINDOW", self.close)

        self.info = tk.Label(root, font=("Microsoft YaHei", 13), anchor="w")
        self.info.pack(fill="x", padx=14, pady=(10, 2))
        panel = tk.Frame(root, bg="#222")
        panel.pack(fill="both", expand=True, padx=14, pady=8)
        self.source_label = tk.Label(panel, bg="#222")
        self.source_label.pack(side="left", fill="both", expand=True, padx=8, pady=8)
        self.split_label = tk.Label(panel, bg="#222")
        self.split_label.pack(side="right", fill="both", expand=True, padx=8, pady=8)
        self.entry_info = tk.Label(root, font=("Consolas", 28, "bold"), fg="#0a67a3")
        self.entry_info.pack(pady=2)
        help_text = (
            "输入两位数字，Enter/空格确认    P 纯图案    X 空白/误检    "
            "S 暂时跳过    U 撤销\n"
            "←/→ 浏览    [ / ] 调整中线间隔    Backspace 删除    Q 保存退出"
        )
        tk.Label(root, text=help_text, font=("Microsoft YaHei", 11)).pack(pady=(2, 10))
        root.bind("<Key>", self.on_key)
        root.bind("<Configure>", lambda _event: root.after_idle(self.render))

        if not self.paths:
            messagebox.showerror("没有图片", "指定目录中没有找到支持的图片。")
            root.after(0, root.destroy)
        else:
            self.render()

    def _key(self, path: Path) -> str:
        try:
            return path.relative_to(self.csv_path.parent).as_posix()
        except ValueError:
            return str(path)

    def _load_records(self) -> dict[str, Record]:
        if not self.csv_path.exists():
            return {}
        records = {}
        with self.csv_path.open("r", newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                key = row.get("path", "")
                if not key:
                    continue
                try:
                    gap = int(row.get("split_gap", self.gap))
                except ValueError:
                    gap = self.gap
                records[key] = Record(row.get("label", ""),
                                      row.get("status", "unlabeled"), max(0, gap))
        return records

    def _first_unlabeled(self) -> int:
        for index, path in enumerate(self.paths):
            if self.records.get(self._key(path), Record()).status == "unlabeled":
                return index
        return 0

    def _save(self) -> None:
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.csv_path.with_suffix(self.csv_path.suffix + ".tmp")
        with temp.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
            writer.writeheader()
            for path in self.paths:
                key = self._key(path)
                rec = self.records.get(key, Record(split_gap=self.gap))
                writer.writerow({"path": key, "label": rec.label,
                                 "status": rec.status, "split_gap": rec.split_gap})
        os.replace(temp, self.csv_path)

    @staticmethod
    def _fit(image: Image.Image, width: int, height: int) -> Image.Image:
        copy = image.copy()
        copy.thumbnail((max(80, width), max(80, height)), Image.Resampling.LANCZOS)
        return copy

    @staticmethod
    def _split_preview(image: Image.Image, gap: int) -> Image.Image:
        width, height = image.size
        middle = width // 2
        gap = min(gap, max(0, middle - 1))
        left = image.crop((0, 0, max(1, middle - gap), height)).convert("RGB")
        right = image.crop((min(width - 1, middle + gap), 0, width, height)).convert("RGB")
        side = max(left.width, right.width, height)
        left = ImageOps.pad(left, (side, side), color="white")
        right = ImageOps.pad(right, (side, side), color="white")
        result = Image.new("RGB", (side * 2 + 12, side), "#d03030")
        result.paste(left, (0, 0))
        result.paste(right, (side + 12, 0))
        return result

    def render(self) -> None:
        if not self.paths or not self.root.winfo_exists():
            return
        path = self.paths[self.index]
        key = self._key(path)
        rec = self.records.get(key, Record(split_gap=self.gap))
        counts = {status: sum(r.status == status for r in self.records.values())
                  for status in ("labeled", "pattern", "invalid", "skipped")}
        self.info.config(
            text=(f"{self.index + 1}/{len(self.paths)}   数字 {counts['labeled']}   "
                  f"图案 {counts['pattern']}   无效 {counts['invalid']}   "
                  f"跳过 {counts['skipped']}   间隔 ±{self.gap}px\n{key}"))
        shown = self.typed or (rec.label if rec.status == "labeled" else "__")
        self.entry_info.config(text=f"标签: {shown}    状态: {rec.status}")
        try:
            with Image.open(path) as opened:
                image = ImageOps.exif_transpose(opened).convert("RGB")
            preview = self._split_preview(image, self.gap)
            panel_w = max(350, self.root.winfo_width() // 2 - 50)
            panel_h = max(350, self.root.winfo_height() - 180)
            shown_images = [self._fit(image, panel_w, panel_h),
                            self._fit(preview, panel_w, panel_h)]
            self.photo_refs = [ImageTk.PhotoImage(item) for item in shown_images]
            self.source_label.config(image=self.photo_refs[0])
            self.split_label.config(image=self.photo_refs[1])
        except Exception as exc:
            self.source_label.config(image="", text=f"读取失败\n{exc}", fg="white")

    def _record(self, label: str, status: str) -> None:
        key = self._key(self.paths[self.index])
        old = self.records.get(key, Record(split_gap=self.gap))
        self.undo_stack.append((key, Record(old.label, old.status, old.split_gap)))
        self.records[key] = Record(label, status, self.gap)
        self.typed = ""
        self._save()
        self._advance()

    def _advance(self) -> None:
        for offset in range(1, len(self.paths) + 1):
            candidate = (self.index + offset) % len(self.paths)
            key = self._key(self.paths[candidate])
            if self.records.get(key, Record()).status == "unlabeled":
                self.index = candidate
                self.render()
                return
        self.render()
        messagebox.showinfo("标注完成", "当前图片集已全部处理，可按 Q 退出。")

    def undo(self) -> None:
        if not self.undo_stack:
            return
        key, old = self.undo_stack.pop()
        self.records[key] = old
        self.index = next(i for i, path in enumerate(self.paths) if self._key(path) == key)
        self.typed = ""
        self._save()
        self.render()

    def on_key(self, event: tk.Event) -> None:
        key = event.keysym.lower()
        if event.char and event.char.isdigit() and len(self.typed) < 2:
            self.typed += event.char
        elif key == "backspace":
            self.typed = self.typed[:-1]
        elif key in {"return", "space"}:
            if len(self.typed) != 2:
                self.root.bell()
            else:
                self._record(self.typed, "labeled")
                return
        elif key == "p":
            self._record("", "pattern")
            return
        elif key == "x":
            self._record("", "invalid")
            return
        elif key == "s":
            self._record("", "skipped")
            return
        elif key == "u" or (key == "z" and event.state & 0x4):
            self.undo()
            return
        elif key == "left":
            self.index = max(0, self.index - 1)
            self.typed = ""
        elif key == "right":
            self.index = min(len(self.paths) - 1, self.index + 1)
            self.typed = ""
        elif event.char == "[":
            self.gap = max(0, self.gap - 1)
        elif event.char == "]":
            self.gap += 1
        elif key in {"q", "escape"}:
            self.close()
            return
        self.render()

    def close(self) -> None:
        if self.paths:
            self._save()
        self.root.destroy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", help="directory containing rectified candidates")
    parser.add_argument("--output", default="digit_labels_candidates.csv")
    parser.add_argument("--gap", type=int, default=3)
    args = parser.parse_args()
    root = tk.Tk()
    CandidateAnnotator(root, Path(args.directory), Path(args.output), args.gap)
    root.mainloop()


if __name__ == "__main__":
    main()
