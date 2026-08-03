"""基于二值图黑色像素比例的曝光标定与离线筛选工具。

本文件完全独立，不依赖 detect_well.py、detect_num.py 或实时曝光控制器。

示例：
    python3 black_ratio_exposure.py stats binary_result_1/binary_result_1 \
        --input-type binary --config black_ratio_limits.json \
        --csv black_ratio_stats.csv

    python3 black_ratio_exposure.py screen 新拍摄图片目录 \
        --input-type color --config black_ratio_limits.json \
        --csv exposure_screen.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
OVEREXPOSED = "OVEREXPOSED"
NORMAL = "NORMAL"
UNDEREXPOSED = "UNDEREXPOSED"


@dataclass
class BlackRatioLimits:
    """黑色像素比例的有效范围及其统计信息。"""

    min_black_ratio: float
    max_black_ratio: float
    absolute_min_black_ratio: float
    absolute_max_black_ratio: float
    lower_percentile: float
    upper_percentile: float
    margin_ratio: float
    black_threshold: int
    input_type: str
    sample_count: int


@dataclass
class ImageRatioResult:
    """单张图片的黑色像素统计结果。"""

    path: str
    black_pixels: int
    valid_pixels: int
    black_ratio: float
    state: str = ""


def iter_images(path: Path) -> Iterable[Path]:
    """按文件名顺序遍历单张图片或目录中的全部图片。"""

    if path.is_file():
        if path.suffix.lower() in IMAGE_SUFFIXES:
            yield path
        return
    if not path.is_dir():
        raise FileNotFoundError("input path does not exist: {}".format(path))
    for image_path in sorted(path.rglob("*")):
        if image_path.is_file() and image_path.suffix.lower() in IMAGE_SUFFIXES:
            yield image_path


def binarize_color_image(image: np.ndarray) -> np.ndarray:
    """采用与数字识别一致的灰度、模糊和 Otsu 二值化流程。"""

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    return binary


def normalize_binary_image(image: np.ndarray, black_threshold: int) -> np.ndarray:
    """把已有二值图片重新归一化，消除 JPEG 压缩产生的近黑和近白灰度。"""

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    return np.where(gray <= black_threshold, 0, 255).astype(np.uint8)


def crop_margin(image: np.ndarray, margin_ratio: float) -> np.ndarray:
    """去掉透视裁剪图外围，避免黑边影响黑色像素统计。"""

    if not 0.0 <= margin_ratio < 0.5:
        raise ValueError("margin ratio must be in [0, 0.5)")
    height, width = image.shape[:2]
    margin_x = int(round(width * margin_ratio))
    margin_y = int(round(height * margin_ratio))
    if width - 2 * margin_x <= 0 or height - 2 * margin_y <= 0:
        raise ValueError("margin removes the entire image")
    return image[margin_y : height - margin_y, margin_x : width - margin_x]


def measure_black_ratio(
    image: np.ndarray,
    input_type: str,
    margin_ratio: float = 0.06,
    black_threshold: int = 127,
) -> Tuple[int, int, float]:
    """计算去掉外围区域后的黑色像素数量和比例。"""

    if image is None or image.size == 0:
        raise ValueError("image is empty")
    if input_type == "color":
        binary = binarize_color_image(image)
    elif input_type == "binary":
        binary = normalize_binary_image(image, black_threshold)
    else:
        raise ValueError("input type must be 'color' or 'binary'")

    valid_region = crop_margin(binary, margin_ratio)
    black_pixels = int(np.count_nonzero(valid_region == 0))
    valid_pixels = int(valid_region.size)
    black_ratio = float(black_pixels) / float(valid_pixels)
    return black_pixels, valid_pixels, black_ratio


def analyze_images(
    input_path: Path,
    input_type: str,
    margin_ratio: float,
    black_threshold: int,
) -> Tuple[List[ImageRatioResult], List[Tuple[str, str]]]:
    """统计路径中的图片，并单独返回读取失败的文件。"""

    results: List[ImageRatioResult] = []
    errors: List[Tuple[str, str]] = []
    for image_path in iter_images(input_path):
        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            errors.append((str(image_path), "image read failed"))
            continue
        try:
            black_pixels, valid_pixels, black_ratio = measure_black_ratio(
                image,
                input_type=input_type,
                margin_ratio=margin_ratio,
                black_threshold=black_threshold,
            )
        except Exception as exc:
            errors.append((str(image_path), str(exc)))
            continue
        results.append(
            ImageRatioResult(
                path=str(image_path),
                black_pixels=black_pixels,
                valid_pixels=valid_pixels,
                black_ratio=black_ratio,
            )
        )
    return results, errors


def calculate_limits(
    results: Sequence[ImageRatioResult],
    lower_percentile: float,
    upper_percentile: float,
    margin_ratio: float,
    black_threshold: int,
    input_type: str,
) -> BlackRatioLimits:
    """计算绝对最小/最大值以及用于筛选的百分位范围。"""

    if not results:
        raise ValueError("no valid images were found")
    if not 0.0 <= lower_percentile < upper_percentile <= 100.0:
        raise ValueError("percentiles must satisfy 0 <= lower < upper <= 100")

    ratios = np.asarray([item.black_ratio for item in results], dtype=np.float64)
    return BlackRatioLimits(
        min_black_ratio=float(np.percentile(ratios, lower_percentile)),
        max_black_ratio=float(np.percentile(ratios, upper_percentile)),
        absolute_min_black_ratio=float(np.min(ratios)),
        absolute_max_black_ratio=float(np.max(ratios)),
        lower_percentile=float(lower_percentile),
        upper_percentile=float(upper_percentile),
        margin_ratio=float(margin_ratio),
        black_threshold=int(black_threshold),
        input_type=input_type,
        sample_count=len(results),
    )


def classify_black_ratio(
    black_ratio: float,
    min_black_ratio: float,
    max_black_ratio: float,
) -> str:
    """按照标定范围判断过曝、正常或过暗。"""

    if black_ratio < min_black_ratio:
        return OVEREXPOSED
    if black_ratio > max_black_ratio:
        return UNDEREXPOSED
    return NORMAL


def save_limits(path: Path, limits: BlackRatioLimits) -> None:
    """保存标定结果，供后续筛选程序直接读取。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(asdict(limits), file, ensure_ascii=False, indent=2)


def load_limits(path: Path) -> BlackRatioLimits:
    """读取黑色像素比例标定配置。"""

    with path.open("r", encoding="utf-8") as file:
        values = json.load(file)
    return BlackRatioLimits(**values)


def save_csv(
    path: Path,
    results: Sequence[ImageRatioResult],
    errors: Sequence[Tuple[str, str]],
) -> None:
    """将逐图统计结果和错误信息写入 CSV。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(
            ["path", "black_pixels", "valid_pixels", "black_ratio", "state", "error"]
        )
        for item in results:
            writer.writerow(
                [
                    item.path,
                    item.black_pixels,
                    item.valid_pixels,
                    "{:.8f}".format(item.black_ratio),
                    item.state,
                    "",
                ]
            )
        for image_path, error in errors:
            writer.writerow([image_path, "", "", "", "READ_ERROR", error])


def print_summary(results: Sequence[ImageRatioResult], errors: Sequence[Tuple[str, str]]) -> None:
    """在终端输出分类数量汇总。"""

    counts = {OVEREXPOSED: 0, NORMAL: 0, UNDEREXPOSED: 0}
    for item in results:
        if item.state in counts:
            counts[item.state] += 1
    print("valid_images={}".format(len(results)))
    print("read_errors={}".format(len(errors)))
    if any(item.state for item in results):
        print("overexposed={}".format(counts[OVEREXPOSED]))
        print("normal={}".format(counts[NORMAL]))
        print("underexposed={}".format(counts[UNDEREXPOSED]))


def run_stats(args: argparse.Namespace) -> int:
    """执行正常样本标定。"""

    results, errors = analyze_images(
        args.input,
        input_type=args.input_type,
        margin_ratio=args.margin,
        black_threshold=args.black_threshold,
    )
    limits = calculate_limits(
        results,
        lower_percentile=args.lower_percentile,
        upper_percentile=args.upper_percentile,
        margin_ratio=args.margin,
        black_threshold=args.black_threshold,
        input_type=args.input_type,
    )
    save_limits(args.config, limits)
    if args.csv is not None:
        save_csv(args.csv, results, errors)

    print("sample_count={}".format(limits.sample_count))
    print("absolute_min_black_ratio={:.8f}".format(limits.absolute_min_black_ratio))
    print("absolute_max_black_ratio={:.8f}".format(limits.absolute_max_black_ratio))
    print("selected_min_black_ratio={:.8f}".format(limits.min_black_ratio))
    print("selected_max_black_ratio={:.8f}".format(limits.max_black_ratio))
    print("config={}".format(args.config))
    print_summary(results, errors)
    return 0


def run_screen(args: argparse.Namespace) -> int:
    """按照已有标定范围筛选新拍摄的图片。"""

    limits = load_limits(args.config)
    input_type = args.input_type or limits.input_type
    margin_ratio = limits.margin_ratio if args.margin is None else args.margin
    black_threshold = (
        limits.black_threshold if args.black_threshold is None else args.black_threshold
    )
    results, errors = analyze_images(
        args.input,
        input_type=input_type,
        margin_ratio=margin_ratio,
        black_threshold=black_threshold,
    )
    for item in results:
        item.state = classify_black_ratio(
            item.black_ratio,
            limits.min_black_ratio,
            limits.max_black_ratio,
        )
    save_csv(args.csv, results, errors)
    print(
        "limits=[{:.8f}, {:.8f}]".format(
            limits.min_black_ratio, limits.max_black_ratio
        )
    )
    print_summary(results, errors)
    print("csv={}".format(args.csv))
    return 0


def build_parser() -> argparse.ArgumentParser:
    """创建命令行参数解析器。"""

    parser = argparse.ArgumentParser(
        description="Calibrate and screen exposure by binary black-pixel ratio"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    stats = subparsers.add_parser("stats", help="calculate black-ratio limits")
    stats.add_argument("input", type=Path, help="normal image file or directory")
    stats.add_argument("--input-type", choices=("binary", "color"), default="binary")
    stats.add_argument("--margin", type=float, default=0.06)
    stats.add_argument("--black-threshold", type=int, default=127)
    stats.add_argument("--lower-percentile", type=float, default=0.0)
    stats.add_argument("--upper-percentile", type=float, default=100.0)
    stats.add_argument("--config", type=Path, default=Path("black_ratio_limits.json"))
    stats.add_argument("--csv", type=Path)
    stats.set_defaults(handler=run_stats)

    screen = subparsers.add_parser("screen", help="screen images with saved limits")
    screen.add_argument("input", type=Path, help="new image file or directory")
    screen.add_argument("--input-type", choices=("binary", "color"))
    screen.add_argument("--margin", type=float)
    screen.add_argument("--black-threshold", type=int)
    screen.add_argument("--config", type=Path, default=Path("black_ratio_limits.json"))
    screen.add_argument("--csv", type=Path, default=Path("exposure_screen.csv"))
    screen.set_defaults(handler=run_screen)
    return parser


def main() -> int:
    """命令行入口。"""

    parser = build_parser()
    args = parser.parse_args()
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
