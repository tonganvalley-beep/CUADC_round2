"""Review deployment binarization and CNN predictions for plate crops.

The input images are the 100x100 perspective crops saved by the flight
program in ``send_to_ground``.  This tool deliberately calls
``DigitRecognizer.binarize_plate`` and ``DigitRecognizer.predict`` so the
reviewed preprocessing is exactly the same implementation as deployment.

It writes three resumable artifacts below ``--output``:

* ``binary/``: the exact binary plates shown to the reviewer;
* ``review.csv``: predictions, confidence and both manual review decisions;
* ``training_labels.csv``: rows compatible with ``train_digit_cnn.py``.

Only plates whose binary image was marked reasonable are exposed as
``status=labeled`` in the training CSV.  Corrected labels are still retained
for unreasonable binaries in review.csv, so they can be regenerated after a
binarization change without asking the reviewer for the true digits again.
"""

from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageOps, ImageTk

from digit_recognizer import DigitRecognizer


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
REVIEW_FIELDS = (
    "path", "binary_path", "binary_status", "predicted",
    "left_confidence", "right_confidence", "confidence_status",
    "cnn_status", "correct_label", "status", "split_gap", "model",
    "updated_at",
)
TRAINING_FIELDS = ("path", "label", "status", "split_gap")


@dataclass
class ReviewRecord:
    path: str
    binary_path: str
    binary_status: str = "unlabeled"
    predicted: str = ""
    left_confidence: float = 0.0
    right_confidence: float = 0.0
    confidence_status: str = "LOW"
    cnn_status: str = "unlabeled"
    correct_label: str = ""
    status: str = "unlabeled"
    split_gap: int = 3
    model: str = ""
    updated_at: str = ""


def natural_key(path: Path) -> tuple[str, int, int | str]:
    try:
        return (str(path.parent).lower(), 0, int(path.stem))
    except ValueError:
        return (str(path.parent).lower(), 1, path.stem.lower())


def collect_images(inputs: list[Path]) -> list[Path]:
    paths: list[Path] = []
    seen: set[Path] = set()
    for item in inputs:
        candidates = item.rglob("*") if item.is_dir() else [item]
        for path in candidates:
            resolved = path.resolve()
            if (resolved.is_file() and resolved.suffix.lower() in IMAGE_SUFFIXES
                    and resolved not in seen):
                paths.append(resolved)
                seen.add(resolved)
    return sorted(paths, key=natural_key)


def read_image(path: Path) -> np.ndarray | None:
    try:
        encoded = np.fromfile(path, dtype=np.uint8)
    except OSError:
        return None
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR)


def write_png(path: Path, image: np.ndarray) -> None:
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError(f"无法编码图片：{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded.tofile(path)


def relative_path(path: Path, base: Path) -> str:
    try:
        return Path(os.path.relpath(path, base)).as_posix()
    except ValueError:
        return str(path)


def load_existing(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return {row["path"]: row for row in csv.DictReader(handle)
                if row.get("path")}


def prepare_records(
    paths: list[Path],
    output_dir: Path,
    recognizer: DigitRecognizer,
    model_path: Path,
    split_gap: int,
    left_threshold: float,
    right_threshold: float,
) -> list[ReviewRecord]:
    review_path = output_dir / "review.csv"
    existing = load_existing(review_path)
    records: list[ReviewRecord] = []
    output_dir.mkdir(parents=True, exist_ok=True)

    for index, source in enumerate(paths, 1):
        image = read_image(source)
        if image is None:
            print(f"[{index}/{len(paths)}] READ_ERROR {source}")
            continue
        plate = cv2.resize(image, (100, 100), interpolation=cv2.INTER_LINEAR)
        binary = DigitRecognizer.binarize_plate(plate, split_gap)
        predicted, _minimum, pair = recognizer.predict(plate)
        predicted = predicted or ""

        dataset = source.parent.parent.name if source.parent.name.lower() == "send_to_ground" else source.parent.name
        binary_name = f"{dataset}_{source.stem}.png"
        binary_path = (output_dir / "binary" / binary_name).resolve()
        write_png(binary_path, binary)

        source_key = relative_path(source, output_dir)
        old = existing.get(source_key, {})
        old_gap = old.get("split_gap", "")
        # A binary-quality decision is only valid for the preprocessing that
        # the reviewer actually saw.
        binary_status = (
            old.get("binary_status", "unlabeled")
            if old_gap == str(split_gap) else "unlabeled"
        )
        correct_label = old.get("correct_label", "")
        if len(correct_label) == 2 and correct_label.isdigit():
            # Ground truth remains valid if a later review run changes models;
            # recompute correctness against the new prediction automatically.
            cnn_status = "correct" if predicted == correct_label else "incorrect"
        else:
            cnn_status = "unlabeled"
            correct_label = ""
        training_status = (
            "labeled"
            if binary_status == "reasonable" and len(correct_label) == 2
               and correct_label.isdigit()
            else ("excluded_binary" if binary_status == "unreasonable" else "unlabeled")
        )
        confidence_status = (
            "PASS" if pair[0] >= left_threshold and pair[1] >= right_threshold
            else "LOW"
        )
        records.append(ReviewRecord(
            path=source_key,
            binary_path=relative_path(binary_path, output_dir),
            binary_status=binary_status,
            predicted=predicted,
            left_confidence=pair[0],
            right_confidence=pair[1],
            confidence_status=confidence_status,
            cnn_status=cnn_status,
            correct_label=correct_label,
            status=training_status,
            split_gap=split_gap,
            model=relative_path(model_path.resolve(), output_dir),
            updated_at=old.get("updated_at", ""),
        ))
        print(
            f"[{index}/{len(paths)}] {dataset}/{source.name} pred={predicted or '--'} "
            f"left={pair[0]:.4f} right={pair[1]:.4f} {confidence_status}"
        )
    return records


class DigitReviewApp:
    def __init__(self, root: tk.Tk, records: list[ReviewRecord], output_dir: Path):
        self.root = root
        self.records = records
        self.output_dir = output_dir.resolve()
        self.index = self._first_incomplete()
        self.photo_refs: list[ImageTk.PhotoImage] = []
        self.undo_stack: list[tuple[int, ReviewRecord]] = []
        self.true_label = tk.StringVar()

        root.title("pic_22 / pic_23 数字牌二值化与 CNN 复核")
        root.geometry("1280x850")
        root.minsize(980, 720)
        root.protocol("WM_DELETE_WINDOW", self.close)

        self.info = ttk.Label(root, font=("Microsoft YaHei UI", 12), anchor="w")
        self.info.pack(fill="x", padx=18, pady=(12, 5))

        image_panel = tk.Frame(root, bg="#20242b")
        image_panel.pack(fill="both", expand=True, padx=18, pady=6)
        original_frame = tk.Frame(image_panel, bg="#20242b")
        original_frame.pack(side="left", fill="both", expand=True, padx=8, pady=8)
        binary_frame = tk.Frame(image_panel, bg="#20242b")
        binary_frame.pack(side="right", fill="both", expand=True, padx=8, pady=8)
        tk.Label(original_frame, text="现场关键点裁剪原图", fg="white", bg="#20242b",
                 font=("Microsoft YaHei UI", 14, "bold")).pack(pady=(2, 7))
        tk.Label(binary_frame, text="当前逻辑二值化结果", fg="white", bg="#20242b",
                 font=("Microsoft YaHei UI", 14, "bold")).pack(pady=(2, 7))
        self.original_label = tk.Label(original_frame, bg="#20242b")
        self.original_label.pack(fill="both", expand=True)
        self.binary_label = tk.Label(binary_frame, bg="#20242b")
        self.binary_label.pack(fill="both", expand=True)

        self.prediction = ttk.Label(root, anchor="center",
                                    font=("Microsoft YaHei UI", 20, "bold"))
        self.prediction.pack(fill="x", padx=18, pady=(6, 5))

        controls = ttk.Frame(root)
        controls.pack(fill="x", padx=18, pady=5)
        binary_box = ttk.LabelFrame(controls, text="1. 二值化是否合理")
        binary_box.pack(side="left", fill="both", expand=True, padx=(0, 7))
        ttk.Button(binary_box, text="合理  [A]", command=lambda: self.set_binary("reasonable")).pack(
            side="left", fill="x", expand=True, padx=8, pady=10)
        ttk.Button(binary_box, text="不合理  [D]", command=lambda: self.set_binary("unreasonable")).pack(
            side="left", fill="x", expand=True, padx=8, pady=10)

        cnn_box = ttk.LabelFrame(controls, text="2. CNN 识别是否正确")
        cnn_box.pack(side="right", fill="both", expand=True, padx=(7, 0))
        ttk.Button(cnn_box, text="正确  [J]", command=self.set_cnn_correct).pack(
            side="left", fill="x", expand=True, padx=8, pady=10)
        ttk.Label(cnn_box, text="错误，正确数字：").pack(side="left", padx=(8, 2))
        self.true_entry = ttk.Entry(cnn_box, textvariable=self.true_label, width=5,
                                    font=("Consolas", 18), justify="center")
        self.true_entry.pack(side="left", padx=3, pady=8)
        ttk.Button(cnn_box, text="提交错误  [Enter]", command=self.set_cnn_incorrect).pack(
            side="left", padx=8, pady=10)

        nav = ttk.Frame(root)
        nav.pack(fill="x", padx=18, pady=(3, 3))
        ttk.Button(nav, text="← 上一张", command=lambda: self.move(-1)).pack(side="left")
        ttk.Button(nav, text="撤销  [U]", command=self.undo).pack(side="left", padx=8)
        ttk.Button(nav, text="下一张 →", command=lambda: self.move(1)).pack(side="left")
        ttk.Button(nav, text="保存并退出  [Q]", command=self.close).pack(side="right")

        ttk.Label(
            root,
            text="快捷键：A 合理｜D 不合理｜J 识别正确｜直接输入两位正确数字后 Enter｜←/→ 浏览｜U 撤销｜Q 退出",
            anchor="center",
        ).pack(fill="x", padx=18, pady=(3, 10))

        root.bind("<Key>", self.on_key)
        root.bind("<Configure>", lambda _event: root.after_idle(self.render))
        self.render()

    def _first_incomplete(self) -> int:
        for index, record in enumerate(self.records):
            if record.binary_status == "unlabeled" or record.cnn_status == "unlabeled":
                return index
        return 0

    def _absolute(self, value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else self.output_dir / path

    @staticmethod
    def _fit(image: Image.Image, max_width: int, max_height: int) -> Image.Image:
        return ImageOps.contain(
            image, (max(120, max_width), max(120, max_height)),
            method=Image.Resampling.NEAREST,
        )

    def render(self) -> None:
        if not self.records or not self.root.winfo_exists():
            return
        record = self.records[self.index]
        complete = sum(
            item.binary_status != "unlabeled" and item.cnn_status != "unlabeled"
            for item in self.records
        )
        binary_bad = sum(item.binary_status == "unreasonable" for item in self.records)
        cnn_bad = sum(item.cnn_status == "incorrect" for item in self.records)
        self.info.config(text=(
            f"{self.index + 1}/{len(self.records)}    已完成 {complete}    "
            f"二值化不合理 {binary_bad}    CNN错误 {cnn_bad}\n"
            f"{record.path}    模型：{record.model}    中缝：±{record.split_gap}px"
        ))
        binary_text = {"reasonable": "合理", "unreasonable": "不合理",
                       "unlabeled": "未标注"}[record.binary_status]
        cnn_text = {"correct": "正确", "incorrect": "错误",
                    "unlabeled": "未标注"}[record.cnn_status]
        confidence_color = "#16803a" if record.confidence_status == "PASS" else "#b24b00"
        corrected = f"    正确数字：{record.correct_label}" if record.correct_label else ""
        self.prediction.config(
            text=(f"CNN 预测：{record.predicted or '--'}    左 {record.left_confidence:.3f}    "
                  f"右 {record.right_confidence:.3f}    {record.confidence_status}\n"
                  f"当前标注：二值化 {binary_text}｜CNN {cnn_text}{corrected}"),
            foreground=confidence_color,
        )
        self.true_label.set(record.correct_label if record.cnn_status == "incorrect" else "")
        try:
            with Image.open(self._absolute(record.path)) as opened:
                original = ImageOps.exif_transpose(opened).convert("RGB")
            with Image.open(self._absolute(record.binary_path)) as opened:
                binary = opened.convert("RGB")
            width = max(400, self.root.winfo_width() // 2 - 70)
            height = max(320, self.root.winfo_height() - 390)
            shown = [self._fit(original, width, height), self._fit(binary, width, height)]
            self.photo_refs = [ImageTk.PhotoImage(image) for image in shown]
            self.original_label.config(image=self.photo_refs[0], text="")
            self.binary_label.config(image=self.photo_refs[1], text="")
        except Exception as exc:
            self.original_label.config(image="", text=f"读取失败\n{exc}", fg="white")

    def _snapshot(self) -> None:
        self.undo_stack.append((self.index, replace(self.records[self.index])))

    def _refresh_training_status(self, record: ReviewRecord) -> None:
        if (record.binary_status == "reasonable" and len(record.correct_label) == 2
                and record.correct_label.isdigit()):
            record.status = "labeled"
        elif record.binary_status == "unreasonable":
            record.status = "excluded_binary"
        else:
            record.status = "unlabeled"
        record.updated_at = datetime.now().astimezone().isoformat(timespec="seconds")

    def set_binary(self, status: str) -> None:
        self._snapshot()
        record = self.records[self.index]
        record.binary_status = status
        self._refresh_training_status(record)
        self.save()
        self._advance_if_complete()

    def set_cnn_correct(self) -> None:
        record = self.records[self.index]
        if len(record.predicted) != 2 or not record.predicted.isdigit():
            messagebox.showwarning("没有有效预测", "当前模型没有给出两位数字，须输入正确数字后提交错误。")
            self.true_entry.focus_set()
            return
        self._snapshot()
        record.cnn_status = "correct"
        record.correct_label = record.predicted
        self._refresh_training_status(record)
        self.save()
        self._advance_if_complete()

    def set_cnn_incorrect(self) -> None:
        value = self.true_label.get().strip()
        if len(value) != 2 or not value.isdigit():
            messagebox.showwarning("请输入正确数字", "请填写两位数字，例如 07，然后再提交。")
            self.true_entry.focus_set()
            self.true_entry.selection_range(0, tk.END)
            return
        self._snapshot()
        record = self.records[self.index]
        record.cnn_status = "incorrect"
        record.correct_label = value
        self._refresh_training_status(record)
        self.save()
        self._advance_if_complete()

    def _advance_if_complete(self) -> None:
        record = self.records[self.index]
        if record.binary_status != "unlabeled" and record.cnn_status != "unlabeled":
            for offset in range(1, len(self.records) + 1):
                candidate = (self.index + offset) % len(self.records)
                item = self.records[candidate]
                if item.binary_status == "unlabeled" or item.cnn_status == "unlabeled":
                    self.index = candidate
                    self.render()
                    return
            self.render()
            messagebox.showinfo("复核完成", "28 张图片已全部复核并自动保存。")
            return
        self.render()

    def move(self, amount: int) -> None:
        self.index = max(0, min(len(self.records) - 1, self.index + amount))
        self.render()

    def undo(self) -> None:
        if not self.undo_stack:
            self.root.bell()
            return
        index, old = self.undo_stack.pop()
        self.index = index
        self.records[index] = old
        self.save()
        self.render()

    def on_key(self, event: tk.Event) -> None:
        key = event.keysym.lower()
        if self.root.focus_get() is self.true_entry:
            if key == "return":
                self.set_cnn_incorrect()
            elif key == "escape":
                self.root.focus_set()
            return
        if event.char and event.char.isdigit():
            self.true_entry.focus_set()
            self.true_label.set(event.char)
            self.true_entry.icursor(tk.END)
        elif key == "a":
            self.set_binary("reasonable")
        elif key == "d":
            self.set_binary("unreasonable")
        elif key == "j":
            self.set_cnn_correct()
        elif key == "left":
            self.move(-1)
        elif key == "right":
            self.move(1)
        elif key == "u":
            self.undo()
        elif key in {"q", "escape"}:
            self.close()

    def save(self) -> None:
        review_path = self.output_dir / "review.csv"
        temp_path = review_path.with_suffix(".csv.tmp")
        with temp_path.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=REVIEW_FIELDS)
            writer.writeheader()
            for record in self.records:
                writer.writerow({
                    "path": record.path,
                    "binary_path": record.binary_path,
                    "binary_status": record.binary_status,
                    "predicted": record.predicted,
                    "left_confidence": f"{record.left_confidence:.6f}",
                    "right_confidence": f"{record.right_confidence:.6f}",
                    "confidence_status": record.confidence_status,
                    "cnn_status": record.cnn_status,
                    "correct_label": record.correct_label,
                    "status": record.status,
                    "split_gap": record.split_gap,
                    "model": record.model,
                    "updated_at": record.updated_at,
                })
        os.replace(temp_path, review_path)

        training_path = self.output_dir / "training_labels.csv"
        training_temp = training_path.with_suffix(".csv.tmp")
        with training_temp.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=TRAINING_FIELDS)
            writer.writeheader()
            for record in self.records:
                writer.writerow({
                    "path": record.binary_path,
                    "label": record.correct_label,
                    "status": record.status,
                    "split_gap": record.split_gap,
                })
        os.replace(training_temp, training_path)

    def close(self) -> None:
        self.save()
        self.root.destroy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "inputs", nargs="*", type=Path,
        default=[Path("pic_22/send_to_ground"), Path("pic_23/send_to_ground")],
        help="send_to_ground 图片或目录",
    )
    parser.add_argument(
        "--model", type=Path,
        default=Path("weights/digit_cnn_v5_pic22_23_shadow_dynamic_finetune_v3.ts"),
        help="现场 CNN TorchScript 权重",
    )
    parser.add_argument("--output", type=Path,
                        default=Path("digit_review_pic22_23_v2_shadow_dynamic"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--split-gap", type=int, default=3)
    parser.add_argument("--left-threshold", type=float, default=0.40)
    parser.add_argument("--right-threshold", type=float, default=0.60)
    parser.add_argument("--prepare-only", action="store_true",
                        help="只生成二值图和推理 CSV，不打开窗口")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = collect_images(args.inputs)
    if not paths:
        raise SystemExit("没有找到 send_to_ground 图片")
    if not args.model.is_file():
        raise SystemExit(f"模型不存在：{args.model}")

    output_dir = args.output.resolve()
    # Set confidence to zero so the raw top-1 digits remain visible even when
    # deployment would reject the plate as LOW confidence.
    recognizer = DigitRecognizer(
        str(args.model), device=args.device, confidence=0.0,
        split_gap=args.split_gap,
    )
    records = prepare_records(
        paths, output_dir, recognizer, args.model, args.split_gap,
        args.left_threshold, args.right_threshold,
    )
    if not records:
        raise SystemExit("图片均读取失败")

    # Persist predictions before opening the UI, so a window/display failure
    # never loses the completed batch inference.
    if args.prepare_only:
        root = tk.Tcl()
        app = DigitReviewApp.__new__(DigitReviewApp)
        app.records = records
        app.output_dir = output_dir
        app.save()
    else:
        root = tk.Tk()
        app = DigitReviewApp(root, records, output_dir)
        app.save()
        root.mainloop()

    passed = sum(record.confidence_status == "PASS" for record in records)
    print(f"summary: images={len(records)} confidence_pass={passed} output={output_dir}")


if __name__ == "__main__":
    main()
