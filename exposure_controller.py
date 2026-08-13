"""相机曝光评估与闭环控制。

检测程序提供整帧，并在检测成功时补充数字牌裁剪图。本模块独立负责曝光统计、
滞回判断和曝光时间决策；相机线程仍然是唯一操作 GStreamer 相机对象的线程。
"""

from __future__ import annotations

import argparse
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, Optional, Tuple

import cv2
import numpy as np


UNDEREXPOSED = "UNDEREXPOSED"
NORMAL = "NORMAL"
OVEREXPOSED = "OVEREXPOSED"


@dataclass
class ExposureResult:
    state: str
    source: str
    score: float
    metrics: Dict[str, float]
    reason: str


@dataclass
class ExposureConfig:
    initial_exposure_ns: int = 5_000_000       # 初始曝光时间，单位：纳秒（5 ms）
    min_exposure_ns: int = 1_000_000           # 最小曝光时间，单位：纳秒（1 ms）
    max_exposure_ns: int = 10_000_000          # 最大曝光时间，单位：纳秒（10 ms）
    decrease_factor: float = 0.60               # 过曝时快速降到当前曝光的 60%
    increase_step_ns: int = 500_000             # 欠曝时每次增加的固定曝光时间，单位：纳秒（0.5 ms）
    plate_confirmations: int = 2                # 欠曝牌面连续异常多少次后才允许调节曝光
    plate_overexposed_confirmations: int = 1    # 裁剪图一旦确认过曝就立即请求降曝光
    frame_confirmations: int = 3                # 整帧连续异常多少次后才允许调节曝光
    plate_hold_seconds: float = 2.0             # 检测到数字牌后，暂时停用整帧兜底判断的秒数
    frame_interval_seconds: float = 0.50        # 两次整帧曝光评估之间的最小间隔，单位：秒
    adjustment_cooldown_seconds: float = 0.15   # 两次曝光调节之间的最小间隔，单位：秒
    settle_seconds: float = 0.20                # 修改曝光后等待新曝光画面稳定的时间
    log_interval_seconds: float = 2.0           # 两次曝光状态日志输出之间的最小间隔，单位：秒

    @classmethod
    def from_env(cls, initial_exposure_ns: int = 5_000_000) -> "ExposureConfig":
        """从环境变量读取可调参数，不与任何一份检测程序耦合。"""

        return cls(
            initial_exposure_ns=int(
                os.getenv("CAMERA_EXPOSURE_INITIAL_NS", str(initial_exposure_ns))
            ),
            min_exposure_ns=int(os.getenv("CAMERA_EXPOSURE_MIN_NS", "1000000")),
            max_exposure_ns=int(os.getenv("CAMERA_EXPOSURE_MAX_NS", "10000000")),
            decrease_factor=float(os.getenv("CAMERA_EXPOSURE_DOWN_FACTOR", "0.60")),
            increase_step_ns=int(
                os.getenv("CAMERA_EXPOSURE_UP_STEP_NS", "500000")
            ),
            plate_confirmations=int(
                os.getenv("CAMERA_EXPOSURE_PLATE_CONFIRMATIONS", "2")
            ),
            plate_overexposed_confirmations=int(
                os.getenv("CAMERA_EXPOSURE_PLATE_OVER_CONFIRMATIONS", "1")
            ),
            frame_confirmations=int(
                os.getenv("CAMERA_EXPOSURE_FRAME_CONFIRMATIONS", "3")
            ),
            plate_hold_seconds=float(
                os.getenv("CAMERA_EXPOSURE_PLATE_HOLD_SECONDS", "2.0")
            ),
            frame_interval_seconds=float(
                os.getenv("CAMERA_EXPOSURE_FRAME_INTERVAL_SECONDS", "0.50")
            ),
            adjustment_cooldown_seconds=float(
                os.getenv("CAMERA_EXPOSURE_COOLDOWN_SECONDS", "0.15")
            ),
            settle_seconds=float(
                os.getenv("CAMERA_EXPOSURE_SETTLE_SECONDS", "0.20")
            ),
            log_interval_seconds=float(
                os.getenv("CAMERA_EXPOSURE_LOG_INTERVAL_SECONDS", "2.0")
            ),
        )


def _center_crop(image: np.ndarray, margin_ratio: float) -> np.ndarray:
    h, w = image.shape[:2]
    mx = min(max(int(w * margin_ratio), 0), max(0, w // 2 - 1))
    my = min(max(int(h * margin_ratio), 0), max(0, h // 2 - 1))
    return image[my : h - my, mx : w - mx]


def prepare_exposure_views(
    image: np.ndarray, source: str = "plate"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """返回判定实际使用的 ROI、灰度图、Otsu 二值图和阈值。"""

    if image is None or image.size == 0:
        raise ValueError("exposure image is empty")
    if source not in ("plate", "frame"):
        raise ValueError("source must be 'plate' or 'frame'")

    margin = 0.06 if source == "plate" else 0.15
    roi = _center_crop(image, margin)
    if roi.shape[1] > 640:
        scale = 640.0 / float(roi.shape[1])
        roi = cv2.resize(
            roi,
            (640, max(1, int(round(roi.shape[0] * scale)))),
            interpolation=cv2.INTER_AREA,
        )
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi.copy()
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    otsu_threshold, binary = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return roi, gray, binary, float(otsu_threshold)


def _metrics(image: np.ndarray, source: str) -> Dict[str, float]:
    if image is None or image.size == 0:
        raise ValueError("exposure image is empty")

    # 忽略透视裁剪边缘，并复用可视化工具所展示的同一套灰度/二值化步骤。
    roi, gray, binary, otsu_threshold = prepare_exposure_views(image, source)
    if roi.ndim == 3:
        # 灰度会掩盖单个/多个颜色通道已经撞到传感器上限的情况。例如现场图中的
        # 粉白地面，B、R 通道已裁切，但换算后的灰度仍可能不到 245。
        clipped_channels = np.count_nonzero(roi >= 250, axis=2)
        channel_clip_ratio = float(np.mean(clipped_channels >= 1))
        multi_channel_clip_ratio = float(np.mean(clipped_channels >= 2))
    else:
        channel_clip_ratio = float(np.mean(gray >= 250))
        multi_channel_clip_ratio = channel_clip_ratio

    p05, p10, p50, p90, p95, p99 = np.percentile(
        gray, [5, 10, 50, 90, 95, 99]
    )
    bright_ratio = float(np.mean(gray >= 245))
    white_ratio = float(np.mean(gray >= 250))
    dark_ratio = float(np.mean(gray <= 20))
    robust_contrast = float(p90 - p10)

    # 全局比例会把“半幅已经烧白、半幅仍有纹理”的航拍图平均掉。把中心有效区
    # 分为 4x3 块，记录高光在空间上覆盖了多少块；这样大面积裁切可以触发兜底，
    # 一条白色道路标线或一小块反光不会单独触发。
    highlight_mask = gray >= 235
    tile_ratios = []
    for y_indices in np.array_split(np.arange(gray.shape[0]), 3):
        for x_indices in np.array_split(np.arange(gray.shape[1]), 4):
            if y_indices.size and x_indices.size:
                tile = highlight_mask[np.ix_(y_indices, x_indices)]
                tile_ratios.append(float(np.mean(tile)))
    highlight_tile_fraction = float(
        np.mean(np.asarray(tile_ratios) >= 0.20)
    ) if tile_ratios else 0.0
    max_tile_highlight_ratio = max(tile_ratios, default=0.0)

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)#本函数是计算梯度，找出边缘，类似canny，比canny轻量化
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gradient = cv2.magnitude(gx, gy) #梯度模
    edge_p90 = float(np.percentile(gradient, 90)) #找出梯度的90%分位数

    foreground_ratio = float(np.mean(binary == 0))

    return {
        "p05": float(p05),
        "p50": float(p50),
        "p90": float(p90),
        "p95": float(p95),
        "p99": float(p99),
        "bright_ratio": bright_ratio,
        "white_ratio": white_ratio,
        "channel_clip_ratio": channel_clip_ratio,
        "multi_channel_clip_ratio": multi_channel_clip_ratio,
        "highlight_tile_fraction": highlight_tile_fraction,
        "max_tile_highlight_ratio": max_tile_highlight_ratio,
        "dark_ratio": dark_ratio,
        "contrast": robust_contrast,
        "edge_p90": edge_p90,
        "foreground_ratio": foreground_ratio,
        "otsu_threshold": otsu_threshold,
    }


def evaluate_exposure(image: np.ndarray, source: str = "plate") -> ExposureResult:
    """判断一张数字牌裁剪图或整帧兜底区域的曝光状态。

    数字牌本身是白底，因此不能仅凭高亮像素判断过曝。只有发生严重高光裁切，
    或者高光裁切同时伴随对比度、边缘、二值化前景丢失时，才判定牌面过曝。
    """

    if source not in ("plate", "frame"):
        raise ValueError("source must be 'plate' or 'frame'")

    m = _metrics(image, source)
    if source == "plate":
        lost_detail = (
            m["contrast"] < 32.0
            or m["edge_p90"] < 22.0
            or m["foreground_ratio"] < 0.012
        )
        severe_clipping = m["bright_ratio"] >= 0.55
        clipped_and_flat = m["bright_ratio"] >= 0.20 and lost_detail
        bright_and_flat = m["p50"] >= 232.0 and m["contrast"] < 42.0
        underexposed = m["p50"] <= 48.0 and m["dark_ratio"] >= 0.35

        if severe_clipping or clipped_and_flat or bright_and_flat:
            score = min(
                1.0,
                0.55 * m["bright_ratio"]
                + 0.25 * max(0.0, (m["p50"] - 210.0) / 45.0)
                + 0.20 * max(0.0, (42.0 - m["contrast"]) / 42.0),
            )
            reason = "plate highlights clipped and useful digit detail is reduced"
            return ExposureResult(OVEREXPOSED, source, score, m, reason)
        if underexposed:
            score = min(1.0, 0.6 * m["dark_ratio"] + 0.4 * (48.0 - m["p50"]) / 48.0)
            return ExposureResult(
                UNDEREXPOSED, source, score, m, "plate shadows dominate"
            )
    else:
        # 检测不到天井时，直接依据整帧判断。现场过曝并不总是整幅纯白：常见
        # 情况是多个颜色通道先裁切，或半幅地面烧白而另一半道路仍保留纹理。
        # 三组判据分别覆盖均匀过曝、颜色通道裁切和空间上成片的高光；最后仍由
        # frame_confirmations 做连续帧确认，避免瞬时反光引起曝光振荡。
        widespread_clipping = (
            m["bright_ratio"] >= 0.18
            and m["p50"] >= 175.0
            and m["p90"] >= 245.0
        )
        channel_clipping = (
            m["multi_channel_clip_ratio"] >= 0.12
            and m["p50"] >= 175.0
        )
        spatial_clipping = (
            m["highlight_tile_fraction"] >= 0.25
            and m["p50"] >= 165.0
            and m["p90"] >= 240.0
        )
        # 即使 JPEG/ISP 把真正的 255 裁切压回 230 左右，只要画面主体（中位数）
        # 已整体抬高且亮部仍接近上限，也按“泛白”处理。600+ 现场图主要落在此类。
        washed_frame = m["p50"] >= 185.0 and m["p90"] >= 230.0
        if (
            widespread_clipping
            or channel_clipping
            or spatial_clipping
            or washed_frame
        ):
            score = min(
                1.0,
                0.35 * m["bright_ratio"]
                + 0.30 * m["multi_channel_clip_ratio"]
                + 0.20 * m["highlight_tile_fraction"]
                + 0.15 * max(0.0, (m["p50"] - 165.0) / 90.0),
            )
            return ExposureResult(
                OVEREXPOSED,
                source,
                score,
                m,
                "frame has widespread or spatially persistent highlight clipping",
            )
        if m["p50"] <= 38.0 and m["dark_ratio"] >= 0.50:
            return ExposureResult(
                UNDEREXPOSED,
                source,
                min(1.0, m["dark_ratio"]),
                m,
                "central frame region is predominantly dark",
            )

    return ExposureResult(NORMAL, source, 0.0, m, "exposure is inside the hold band")


class ExposureController:
    """位于推理线程和相机线程之间的线程安全曝光决策层。"""

    def __init__(
        self,
        config: Optional[ExposureConfig] = None,
        logger: Optional[Callable[[str], None]] = print,
    ) -> None:
        self.config = config or ExposureConfig.from_env()
        if self.config.min_exposure_ns > self.config.max_exposure_ns:
            raise ValueError("minimum exposure exceeds maximum exposure")
        if self.config.increase_step_ns <= 0:
            raise ValueError("exposure increase step must be positive")
        if not 0.0 < self.config.decrease_factor < 1.0:
            raise ValueError("exposure decrease factor must be between 0 and 1")
        if self.config.plate_overexposed_confirmations <= 0:
            raise ValueError("plate overexposure confirmations must be positive")

        self.current_exposure_ns = int(
            np.clip(
                self.config.initial_exposure_ns,
                self.config.min_exposure_ns,
                self.config.max_exposure_ns,
            )
        )
        self._pending_exposure_ns: Optional[int] = None
        self._lock = threading.Lock()
        self._logger = logger
        self._last_plate_time = 0.0
        self._last_frame_time = 0.0
        self._last_adjustment_time = 0.0
        self._settle_until = 0.0
        self._last_log_time = 0.0
        self._candidate_state = NORMAL
        self._candidate_source = "frame"
        self._candidate_count = 0
        self._exposure_generation = 0

    def snapshot(self) -> Tuple[int, int]:
        """返回相机线程为新捕获帧记录的曝光值和曝光代次。"""

        with self._lock:
            return self.current_exposure_ns, self._exposure_generation

    def observe_plate(
        self,
        plate: np.ndarray,
        exposure_generation: Optional[int] = None,
        captured_at: Optional[float] = None,
    ) -> ExposureResult:
        result = self.observe_plates(
            [plate],
            exposure_generation=exposure_generation,
            captured_at=captured_at,
        )
        if result is None:  # 单元素列表在正常情况下不可能为空。
            raise ValueError("exposure plate is empty")
        return result

    def observe_plates(
        self,
        plates: Iterable[np.ndarray],
        exposure_generation: Optional[int] = None,
        captured_at: Optional[float] = None,
    ) -> Optional[ExposureResult]:
        """把同一帧中的全部数字牌作为一次确认进行处理。

        只要有一块牌过曝就按过曝处理，因为高光中丢失的细节无法恢复。只有所有
        可见数字牌都过暗时才接受过暗结论；结果冲突时保持当前曝光。
        """

        # 推理可能落后相机数帧。曝光已改变后，旧曝光代次的裁剪图不得继续触发
        # 第二次下降；调整后的稳定期内捕获的图也不能作为新曝光的复核证据。
        with self._lock:
            if (
                exposure_generation is not None
                and exposure_generation != self._exposure_generation
            ):
                return None
            if captured_at is not None and captured_at < self._settle_until:
                return None

        results = [evaluate_exposure(plate, "plate") for plate in plates]
        if not results:
            return None
        over = [result for result in results if result.state == OVEREXPOSED]
        under = [result for result in results if result.state == UNDEREXPOSED]
        if over:
            result = max(over, key=lambda item: item.score)
        elif len(under) == len(results):
            result = max(under, key=lambda item: item.score)
        else:
            result = next(
                (item for item in results if item.state == NORMAL), results[0]
            )

        now = time.monotonic()
        with self._lock:
            if (
                exposure_generation is not None
                and exposure_generation != self._exposure_generation
            ):
                return None
            if captured_at is not None and captured_at < self._settle_until:
                return None
            self._last_plate_time = now
            self._consume(result, now, captured_at=captured_at)
        return result

    def observe_frame(self, frame: np.ndarray) -> Optional[ExposureResult]:
        now = time.monotonic()
        with self._lock:
            if now < self._settle_until:
                return None
            if now - self._last_plate_time < self.config.plate_hold_seconds:
                return None
            if now - self._last_frame_time < self.config.frame_interval_seconds:
                return None
            self._last_frame_time = now

        result = evaluate_exposure(frame, "frame")
        with self._lock:
            # 计算整帧指标期间可能刚好收到数字牌结果，此时始终以数字牌证据为准。
            if now - self._last_plate_time >= self.config.plate_hold_seconds:
                self._consume(result, now)
        return result

    def _consume(
        self,
        result: ExposureResult,
        now: float,
        captured_at: Optional[float] = None,
    ) -> None:
        if now < self._settle_until:
            return

        if result.state == NORMAL:
            self._candidate_state = NORMAL
            self._candidate_count = 0
        elif (
            result.state == self._candidate_state
            and result.source == self._candidate_source
        ):
            self._candidate_count += 1
        else:
            self._candidate_state = result.state
            self._candidate_source = result.source
            self._candidate_count = 1

        if self._logger and now - self._last_log_time >= self.config.log_interval_seconds:
            m = result.metrics
            self._logger(
                "[EXPOSURE] source={} state={} exp={}ns p50={:.0f} "
                "bright={:.1%} multi_clip={:.1%} hot_tiles={:.1%} "
                "contrast={:.1f} edge={:.1f} foreground={:.1%} otsu={:.0f}".format(
                    result.source,
                    result.state,
                    self.current_exposure_ns,
                    m["p50"],
                    m["bright_ratio"],
                    m["multi_channel_clip_ratio"],
                    m["highlight_tile_fraction"],
                    m["contrast"],
                    m["edge_p90"],
                    m["foreground_ratio"],
                    m["otsu_threshold"],
                )
            )
            self._last_log_time = now

        if result.source == "plate" and result.state == OVEREXPOSED:
            required = self.config.plate_overexposed_confirmations
        elif result.source == "plate":
            required = self.config.plate_confirmations
        else:
            required = self.config.frame_confirmations
        if self._candidate_count < required:
            return
        if self._pending_exposure_ns is not None:
            return
        if now - self._last_adjustment_time < self.config.adjustment_cooldown_seconds:
            return

        if result.state == OVEREXPOSED:
            requested = int(self.current_exposure_ns * self.config.decrease_factor) #曝光降低
        elif result.state == UNDEREXPOSED:
            requested = self.current_exposure_ns + self.config.increase_step_ns  #曝光增加
        else:
            return

        requested = int(
            np.clip(
                requested,
                self.config.min_exposure_ns,
                self.config.max_exposure_ns,
            )
        )
        if requested == self.current_exposure_ns:
            return

        self._pending_exposure_ns = requested
        self._candidate_count = 0
        if self._logger:
            capture_age_ms = (
                max(0.0, now - captured_at) * 1000.0
                if captured_at is not None
                else -1.0
            )
            self._logger(
                "[EXPOSURE] request generation={} state={} "
                "{}ns->{}ns capture_age_ms={:.1f}".format(
                    self._exposure_generation,
                    result.state,
                    self.current_exposure_ns,
                    requested,
                    capture_age_ms,
                )
            )

    def apply_pending(self, camera) -> Optional[int]:
        """执行队列中的曝光调整请求；此方法只能由相机线程调用。"""

        with self._lock:
            requested = self._pending_exposure_ns
            self._pending_exposure_ns = None
        if requested is None:
            return None

        try:
            camera.set_exposure("{} {}".format(requested, requested))
        except Exception as exc:
            if self._logger:
                self._logger("[EXPOSURE][WARN] camera update failed: {}".format(exc))
            return None

        now = time.monotonic()
        with self._lock:
            previous = self.current_exposure_ns
            self.current_exposure_ns = requested
            self._last_adjustment_time = now
            self._settle_until = now + self.config.settle_seconds
            self._exposure_generation += 1
            self._candidate_state = NORMAL
            self._candidate_count = 0
        if self._logger:
            self._logger(
                "[EXPOSURE] applied {}ns -> {}ns".format(previous, requested)
            )
        return requested


def _main() -> int:
    parser = argparse.ArgumentParser(description="Inspect exposure of saved images")
    parser.add_argument("input", type=Path, help="image file or directory")
    parser.add_argument("--source", choices=("plate", "frame"), default="plate")
    args = parser.parse_args()

    paths = [args.input] if args.input.is_file() else sorted(
        p for p in args.input.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    )
    for path in paths:
        image = cv2.imread(str(path))
        if image is None:
            # OpenCV on Windows may fail to open non-ASCII paths (for example,
            # a directory named "过曝"). imdecode accepts the same bytes after
            # Python has opened the Unicode path itself.
            try:
                encoded = np.fromfile(str(path), dtype=np.uint8)
                image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            except (OSError, ValueError, cv2.error):
                image = None
        if image is None:
            print("{}\tREAD_ERROR".format(path))
            continue
        result = evaluate_exposure(image, args.source)
        m = result.metrics
        print(
            "{}\t{}\tp50={:.0f}\tp90={:.0f}\tbright={:.1%}\t"
            "multi_clip={:.1%}\thot_tiles={:.1%}\tcontrast={:.1f}\t"
            "edge={:.1f}\tforeground={:.1%}".format(
                path,
                result.state,
                m["p50"],
                m["p90"],
                m["bright_ratio"],
                m["multi_channel_clip_ratio"],
                m["highlight_tile_fraction"],
                m["contrast"],
                m["edge_p90"],
                m["foreground_ratio"],
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
