"""Fast two-digit plate annotation tool.

The CSV is saved after every action and can be resumed safely.
Only Pillow and the Python standard library are required.
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


class Annotator:
    def __init__(self, root: tk.Tk, image_dirs: list[Path], csv_path: Path, gap: int):
        self.root = root
        self.csv_path = csv_path.resolve()
        self.gap = max(0, gap)
        self.paths = self._find_images(image_dirs)
        self.records = self._load_records()
        self.index = self._first_unlabeled()
        self.typed = ""
        self.undo_stack: list[tuple[str, Record]] = []
        self.photo_refs: list[ImageTk.PhotoImage] = []

        root.title("CUADC 数字牌快捷标注")
        root.geometry("1050x760")
        root.minsize(820, 620)
        root.protocol("WM_DELETE_WINDOW", self.close)

        self.info = tk.Label(root, font=("Microsoft YaHei", 13), anchor="w")
        self.info.pack(fill="x", padx=14, pady=(10, 2))
        self.image_panel = tk.Frame(root, bg="#222")
        self.image_panel.pack(fill="both", expand=True, padx=14, pady=8)
        self.source_label = tk.Label(self.image_panel, bg="#222")
        self.source_label.pack(side="left", fill="both", expand=True, padx=8, pady=8)
        self.split_label = tk.Label(self.image_panel, bg="#222")
        self.split_label.pack(side="right", fill="both", expand=True, padx=8, pady=8)
        self.entry_info = tk.Label(root, font=("Consolas", 28, "bold"), fg="#0a67a3")
        self.entry_info.pack(pady=2)
        help_text = (
            "直接输入两位数字，Enter/空格确认    Backspace 删除    S 跳过    U 撤销\n"
            "←/→ 浏览    [ / ] 调整中线间隔    Q 保存退出"
        )
        tk.Label(root, text=help_text, font=("Microsoft YaHei", 11)).pack(pady=(2, 10))

        root.bind("<Key>", self.on_key)
        root.bind("<Configure>", lambda _e: self.root.after_idle(self.render))
        if not self.paths:
            messagebox.showerror("没有图片", "指定目录中没有找到支持的图片。")
            root.after(0, root.destroy)
        else:
            self.render()

    @staticmethod
    def _find_images(image_dirs: list[Path]) -> list[Path]:
        found: set[Path] = set()
        for directory in image_dirs:
            directory = directory.resolve()
            if not directory.is_dir():
                continue
            for path in directory.rglob("*"):
                if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                    found.add(path.resolve())
        return sorted(found, key=lambda p: str(p).lower())

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
                key = row.get("path", "")
                if not key:
                    continue
                try:
                    row_gap = int(row.get("split_gap", self.gap))
                except ValueError:
                    row_gap = self.gap
                records[key] = Record(
                    label=row.get("label", ""),
                    status=row.get("status", "unlabeled"),
                    split_gap=max(0, row_gap),
                )
        return records

    def _first_unlabeled(self) -> int:
        for i, path in enumerate(self.paths):
            if self.records.get(self._key(path), Record()).status == "unlabeled":
                return i
        return 0

    def _save(self) -> None:
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self.csv_path.with_suffix(self.csv_path.suffix + ".tmp")
        with temp_path.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
            writer.writeheader()
            for path in self.paths:
                key = self._key(path)
                rec = self.records.get(key, Record(split_gap=self.gap))
                writer.writerow({
                    "path": key,
                    "label": rec.label,
                    "status": rec.status,
                    "split_gap": rec.split_gap,
                })
        os.replace(temp_path, self.csv_path)

    @staticmethod
    def _fit(image: Image.Image, max_w: int, max_h: int) -> Image.Image:
        copy = image.copy()
        copy.thumbnail((max(80, max_w), max(80, max_h)), Image.Resampling.LANCZOS)
        return copy

    def _make_split_preview(self, image: Image.Image, gap: int) -> Image.Image:
        w, h = image.size
        middle = w // 2
        half_gap = min(gap, max(0, middle - 1))
        left = image.crop((0, 0, max(1, middle - half_gap), h)).convert("RGB")
        right = image.crop((min(w - 1, middle + half_gap), 0, w, h)).convert("RGB")
        side = max(left.width, right.width, h)
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
        done = sum(r.status == "labeled" for r in self.records.values())
        skipped = sum(r.status == "skipped" for r in self.records.values())
        self.info.config(
            text=f"{self.index + 1}/{len(self.paths)}   已标 {done}   跳过 {skipped}   "
                 f"间隔 ±{self.gap}px\n{key}"
        )
        shown = self.typed or (rec.label if rec.status == "labeled" else "__")
        self.entry_info.config(text=f"标签: {shown}    状态: {rec.status}")
        try:
            with Image.open(path) as opened:
                image = ImageOps.exif_transpose(opened).convert("RGB")
                preview = self._make_split_preview(image, self.gap)
                panel_w = max(350, self.image_panel.winfo_width() // 2 - 30)
                panel_h = max(350, self.image_panel.winfo_height() - 20)
                src_show = self._fit(image, panel_w, panel_h)
                split_show = self._fit(preview, panel_w, panel_h)
            self.photo_refs = [ImageTk.PhotoImage(src_show), ImageTk.PhotoImage(split_show)]
            self.source_label.config(image=self.photo_refs[0])
            self.split_label.config(image=self.photo_refs[1])
        except Exception as exc:
            self.source_label.config(image="", text=f"读取失败\n{exc}", fg="white")

    def _record(self, label: str, status: str) -> None:
        key = self._key(self.paths[self.index])
        old = self.records.get(key, Record(split_gap=self.gap))
        self.undo_stack.append((key, Record(old.label, old.status, old.split_gap)))
        self.records[key] = Record(label=label, status=status, split_gap=self.gap)
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
        self.index = min(self.index + 1, len(self.paths) - 1)
        self.render()
        messagebox.showinfo("标注完成", "当前图片集已全部处理，可按 Q 退出。")

    def undo(self) -> None:
        if not self.undo_stack:
            return
        key, old = self.undo_stack.pop()
        self.records[key] = old
        for i, path in enumerate(self.paths):
            if self._key(path) == key:
                self.index = i
                break
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
            if len(self.typed) == 2:
                self._record(self.typed, "labeled")
                return
            self.root.bell()
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
        elif key == "q" or key == "escape":
            self.close()
            return
        self.render()

    def close(self) -> None:
        if self.paths:
            self._save()
        self.root.destroy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Label rectified two-digit CUADC plates.")
    parser.add_argument("directories", nargs="*", default=["numbers", "7_17"], help="image directories")
    parser.add_argument("--output", default="digit_labels.csv", help="output CSV path")
    parser.add_argument("--gap", type=int, default=3, help="pixels omitted on each side of center line")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = tk.Tk()
    Annotator(root, [Path(p) for p in args.directories], Path(args.output), args.gap)
    root.mainloop()


if __name__ == "__main__":
    main()
