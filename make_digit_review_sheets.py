"""Build compact color/binary review sheets for rectified digit plates."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def natural_key(path: Path) -> list[object]:
    return [int(part) if part.isdigit() else part.lower()
            for part in re.split(r"(\d+)", path.stem)]


def load_labels(csv_path: Path | None) -> dict[str, tuple[str, str]]:
    if csv_path is None or not csv_path.exists():
        return {}
    with csv_path.open("r", newline="", encoding="utf-8-sig") as handle:
        return {
            Path(row["path"]).stem: (row.get("label", ""), row.get("status", ""))
            for row in csv.DictReader(handle)
            if row.get("path")
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("prepared", type=Path)
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--columns", type=int, default=4)
    parser.add_argument("--rows", type=int, default=4)
    args = parser.parse_args()

    prepared = args.prepared.resolve()
    output = (args.output or prepared / "review_sheets").resolve()
    output.mkdir(parents=True, exist_ok=True)
    colors = {path.stem: path for path in (prepared / "warp_color").glob("*.jpg")}
    binaries = sorted((prepared / "plate_binary").glob("*.png"), key=natural_key)
    labels = load_labels(args.csv.resolve() if args.csv else None)

    cell_w, cell_h = 300, 176
    per_page = args.columns * args.rows
    font = ImageFont.load_default(size=18)
    small_font = ImageFont.load_default(size=14)
    for page_index in range(0, len(binaries), per_page):
        page_items = binaries[page_index:page_index + per_page]
        sheet = Image.new("RGB", (cell_w * args.columns, cell_h * args.rows), "white")
        draw = ImageDraw.Draw(sheet)
        for index, binary_path in enumerate(page_items):
            row, column = divmod(index, args.columns)
            x, y = column * cell_w, row * cell_h
            stem = binary_path.stem
            label, status = labels.get(stem, ("", ""))
            title = stem
            if status:
                title += f"  [{label if status == 'labeled' else status}]"
            draw.rectangle((x, y, x + cell_w - 1, y + cell_h - 1), outline="#888")
            draw.text((x + 10, y + 4), title, fill="black", font=font)
            with Image.open(colors[stem]) as opened:
                color = opened.convert("RGB").resize((125, 125), Image.Resampling.NEAREST)
            with Image.open(binary_path) as opened:
                binary = opened.convert("RGB").resize((125, 125), Image.Resampling.NEAREST)
            sheet.paste(color, (x + 10, y + 34))
            sheet.paste(binary, (x + 158, y + 34))
            draw.text((x + 10, y + 159), "color", fill="#555", font=small_font)
            draw.text((x + 158, y + 159), "binary", fill="#555", font=small_font)
        page_number = page_index // per_page + 1
        sheet.save(output / f"review_{page_number:02d}.png")
    print(f"plates={len(binaries)} sheets={(len(binaries) + per_page - 1) // per_page} output={output}")


if __name__ == "__main__":
    main()
