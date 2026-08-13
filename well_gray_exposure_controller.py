"""Grayscale-only exposure control for detect_well_exposure(3).py.

The classifier in that detector converts every crop from BGR to grayscale
before inference.  This module therefore evaluates only that same grayscale
image.  It deliberately contains no Otsu, color-channel, saturation, edge, or
full-frame exposure rule.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Optional, Tuple

import cv2
import numpy as np


NORMAL = "NORMAL"
OVEREXPOSED = "OVEREXPOSED"


@dataclass(frozen=True)
class ExposureResult:
    state: str
    source: str
    score: float
    metrics: Dict[str, float]
    reason: str


@dataclass
class ExposureConfig:
    initial_exposure_ns: int = 800_000
    min_exposure_ns: int = 100_000
    max_exposure_ns: int = 10_000_000
    decrease_factor: float = 0.60
    overexposed_confirmations: int = 1
    adjustment_cooldown_seconds: float = 0.15
    settle_seconds: float = 0.20
    log_interval_seconds: float = 2.0

    # Learned from the manually labeled pic_07..pic_09/send_to_ground crops.
    center_margin_ratio: float = 0.20
    center_p10_threshold: float = 203.0

    @classmethod
    def from_env(cls, initial_exposure_ns: int = 5_000_000) -> "ExposureConfig":
        return cls(
            initial_exposure_ns=int(
                os.getenv("CAMERA_EXPOSURE_INITIAL_NS", str(initial_exposure_ns))
            ),
            min_exposure_ns=int(os.getenv("CAMERA_EXPOSURE_MIN_NS", "1000000")),
            max_exposure_ns=int(os.getenv("CAMERA_EXPOSURE_MAX_NS", "10000000")),
            decrease_factor=float(os.getenv("CAMERA_EXPOSURE_DOWN_FACTOR", "0.60")),
            overexposed_confirmations=int(
                os.getenv("WELL_GRAY_OVEREXPOSED_CONFIRMATIONS", "1")
            ),
            adjustment_cooldown_seconds=float(
                os.getenv("CAMERA_EXPOSURE_COOLDOWN_SECONDS", "0.15")
            ),
            settle_seconds=float(os.getenv("CAMERA_EXPOSURE_SETTLE_SECONDS", "0.20")),
            log_interval_seconds=float(
                os.getenv("CAMERA_EXPOSURE_LOG_INTERVAL_SECONDS", "2.0")
            ),
            center_margin_ratio=float(
                os.getenv("WELL_GRAY_CENTER_MARGIN_RATIO", "0.20")
            ),
            center_p10_threshold=float(
                os.getenv("WELL_GRAY_OVEREXPOSED_CENTER_P10", "203.0")
            ),
        )


def classifier_gray(image: np.ndarray) -> np.ndarray:
    """Return exactly the single gray channel used to build classifier input."""

    if image is None or image.size == 0:
        raise ValueError("exposure image is empty")
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[2] == 1:
        return image[:, :, 0]
    if image.ndim == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    raise ValueError("expected a grayscale or BGR image")


def center_roi(gray: np.ndarray, margin_ratio: float = 0.20) -> np.ndarray:
    """Keep the central 60% used by the labeled-data threshold."""

    if not 0.0 <= margin_ratio < 0.5:
        raise ValueError("center margin ratio must be in [0, 0.5)")
    height, width = gray.shape[:2]
    margin_x = min(int(round(width * margin_ratio)), max(0, width // 2 - 1))
    margin_y = min(int(round(height * margin_ratio)), max(0, height // 2 - 1))
    return gray[margin_y : height - margin_y, margin_x : width - margin_x]


def evaluate_exposure(
    image: np.ndarray,
    source: str = "plate",
    config: Optional[ExposureConfig] = None,
) -> ExposureResult:
    """Classify a crop using only the center grayscale p10 threshold."""

    if source != "plate":
        raise ValueError("the dedicated grayscale controller only accepts plate crops")
    config = config or ExposureConfig.from_env()
    gray = classifier_gray(image)
    center = center_roi(gray, config.center_margin_ratio)
    center_p10, center_p50, center_p90 = np.percentile(center, (10, 50, 90))
    metrics = {
        "center_p10": float(center_p10),
        "center_p50": float(center_p50),
        "center_p90": float(center_p90),
        "center_mean": float(center.mean()),
        "center_contrast_90_10": float(center_p90 - center_p10),
        "threshold": float(config.center_p10_threshold),
    }
    if center_p10 >= config.center_p10_threshold:
        denominator = max(1.0, 255.0 - config.center_p10_threshold)
        score = min(1.0, 0.5 + 0.5 * (center_p10 - config.center_p10_threshold) / denominator)
        return ExposureResult(
            OVEREXPOSED,
            "plate",
            score,
            metrics,
            "central grayscale p10 reaches the labeled overexposure threshold",
        )
    return ExposureResult(
        NORMAL,
        "plate",
        0.0,
        metrics,
        "central grayscale p10 is below the overexposure threshold",
    )


class ExposureController:
    """Thread-safe, grayscale-only controller matching the detector interface."""

    def __init__(
        self,
        config: Optional[ExposureConfig] = None,
        logger: Optional[Callable[[str], None]] = print,
    ) -> None:
        self.config = config or ExposureConfig.from_env()
        if self.config.min_exposure_ns > self.config.max_exposure_ns:
            raise ValueError("minimum exposure exceeds maximum exposure")
        if not 0.0 < self.config.decrease_factor < 1.0:
            raise ValueError("exposure decrease factor must be between 0 and 1")
        if self.config.overexposed_confirmations <= 0:
            raise ValueError("overexposed confirmations must be positive")
        if not 0.0 <= self.config.center_margin_ratio < 0.5:
            raise ValueError("center margin ratio must be in [0, 0.5)")
        if not 0.0 <= self.config.center_p10_threshold <= 255.0:
            raise ValueError("center p10 threshold must be in [0, 255]")

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
        self._last_adjustment_time = 0.0
        self._settle_until = 0.0
        self._last_log_time = 0.0
        self._overexposed_count = 0
        self._exposure_generation = 0

    def snapshot(self) -> Tuple[int, int]:
        with self._lock:
            return self.current_exposure_ns, self._exposure_generation

    def observe_plate(
        self,
        plate: np.ndarray,
        exposure_generation: Optional[int] = None,
        captured_at: Optional[float] = None,
    ) -> Optional[ExposureResult]:
        return self.observe_plates(
            [plate],
            exposure_generation=exposure_generation,
            captured_at=captured_at,
        )

    def observe_plates(
        self,
        plates: Iterable[np.ndarray],
        exposure_generation: Optional[int] = None,
        captured_at: Optional[float] = None,
    ) -> Optional[ExposureResult]:
        with self._lock:
            if (
                exposure_generation is not None
                and exposure_generation != self._exposure_generation
            ):
                return None
            if captured_at is not None and captured_at < self._settle_until:
                return None

        results = [
            evaluate_exposure(plate, "plate", self.config)
            for plate in plates
            if plate is not None and plate.size
        ]
        if not results:
            return None
        overexposed = [result for result in results if result.state == OVEREXPOSED]
        result = (
            max(overexposed, key=lambda item: item.metrics["center_p10"])
            if overexposed
            else max(results, key=lambda item: item.metrics["center_p10"])
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
            self._consume(result, now, captured_at)
        return result

    def observe_frame(self, _frame: np.ndarray) -> None:
        """Full-frame evaluation is intentionally disabled for this controller."""

        return None

    def _consume(
        self,
        result: ExposureResult,
        now: float,
        captured_at: Optional[float],
    ) -> None:
        if now < self._settle_until:
            return
        if result.state == OVEREXPOSED:
            self._overexposed_count += 1
        else:
            self._overexposed_count = 0

        if self._logger and now - self._last_log_time >= self.config.log_interval_seconds:
            metrics = result.metrics
            self._logger(
                "[WELL_GRAY_EXPOSURE] state={} exp={}ns center_p10={:.1f} "
                "threshold={:.1f} center_p50={:.1f} center_mean={:.1f}".format(
                    result.state,
                    self.current_exposure_ns,
                    metrics["center_p10"],
                    metrics["threshold"],
                    metrics["center_p50"],
                    metrics["center_mean"],
                )
            )
            self._last_log_time = now

        if result.state != OVEREXPOSED:
            return
        if self._overexposed_count < self.config.overexposed_confirmations:
            return
        if self._pending_exposure_ns is not None:
            return
        if now - self._last_adjustment_time < self.config.adjustment_cooldown_seconds:
            return

        requested = int(
            np.clip(
                int(self.current_exposure_ns * self.config.decrease_factor),
                self.config.min_exposure_ns,
                self.config.max_exposure_ns,
            )
        )
        if requested == self.current_exposure_ns:
            return
        self._pending_exposure_ns = requested
        self._overexposed_count = 0
        if self._logger:
            capture_age_ms = (
                max(0.0, now - captured_at) * 1000.0
                if captured_at is not None
                else -1.0
            )
            self._logger(
                "[WELL_GRAY_EXPOSURE] request generation={} {}ns->{}ns "
                "capture_age_ms={:.1f}".format(
                    self._exposure_generation,
                    self.current_exposure_ns,
                    requested,
                    capture_age_ms,
                )
            )

    def apply_pending(self, camera) -> Optional[int]:
        """Apply a queued exposure update from the camera thread only."""

        with self._lock:
            requested = self._pending_exposure_ns
            self._pending_exposure_ns = None
        if requested is None:
            return None
        try:
            camera.set_exposure("{} {}".format(requested, requested))
        except Exception as exc:
            if self._logger:
                self._logger(
                    "[WELL_GRAY_EXPOSURE][WARN] camera update failed: {}".format(exc)
                )
            return None

        now = time.monotonic()
        with self._lock:
            previous = self.current_exposure_ns
            self.current_exposure_ns = requested
            self._last_adjustment_time = now
            self._settle_until = now + self.config.settle_seconds
            self._exposure_generation += 1
            self._overexposed_count = 0
        if self._logger:
            self._logger(
                "[WELL_GRAY_EXPOSURE] applied {}ns -> {}ns".format(
                    previous, requested
                )
            )
        return requested
