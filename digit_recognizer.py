"""TorchScript two-digit recognizer used by detect_num_gps.py."""

from __future__ import annotations

import cv2
import numpy as np
import torch
from typing import Optional, Tuple


class DigitRecognizer:
    def __init__(self, model_path: str, device: str = "cuda", confidence: float = 0.50,
                 split_gap: int = 3, left_confidence: Optional[float] = None,
                 right_confidence: Optional[float] = None):
        self.device = torch.device(device if device == "cpu" or torch.cuda.is_available() else "cpu")
        self.model = torch.jit.load(model_path, map_location=self.device).eval()
        # Keep confidence as a backward-compatible fallback for old callers.
        self.left_confidence = (confidence if left_confidence is None
                                else float(left_confidence))
        self.right_confidence = (confidence if right_confidence is None
                                 else float(right_confidence))
        self.split_gap = max(0, split_gap)

    @staticmethod
    def _binarize_region(region: np.ndarray) -> np.ndarray:
        """Legacy Otsu helper retained for callers comparing old preprocessing."""
        blurred = cv2.GaussianBlur(region, (3, 3), 0)
        _, binary = cv2.threshold(
            blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        return binary

    @classmethod
    def binarize_plate(cls, plate: np.ndarray, split_gap: int = 3) -> np.ndarray:
        """Remove smooth illumination changes, then apply one consistent Otsu.

        The previous left/right Otsu implementation still failed when a shadow
        crossed *within* one digit.  It also converted saturated red/blue well
        backgrounds to dark grayscale, joining them to the digit foreground.
        HSV value keeps bright coloured backgrounds bright.  Dividing it by a
        large Gaussian illumination estimate removes smooth shadows before Otsu
        chooses the foreground threshold.

        ``split_gap`` remains in the signature for backward compatibility; the
        gap is applied only after the actual inter-digit valley is located.
        """
        if plate.ndim == 3:
            value = cv2.cvtColor(plate, cv2.COLOR_BGR2HSV)[:, :, 2]
        else:
            value = plate
        value = cv2.GaussianBlur(value, (3, 3), 0)
        illumination = cv2.GaussianBlur(value, (0, 0), 13)
        normalized = cv2.divide(value, np.maximum(illumination, 1), scale=255)
        _, binary = cv2.threshold(
            normalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        return binary

    @staticmethod
    def find_split(binary: np.ndarray) -> int:
        """Locate the valley between digits instead of assuming image centre.

        Plate borders are mostly horizontal, so the projection ignores the top
        and bottom 12 percent.  A small centre penalty only resolves equally
        empty valleys; it is deliberately too weak to override the ink data.
        """
        if binary.ndim == 3:
            binary = cv2.cvtColor(binary, cv2.COLOR_BGR2GRAY)
        height, width = binary.shape[:2]
        if width < 4:
            return max(1, width // 2)
        y0 = min(height // 2, max(0, round(height * 0.12)))
        y1 = max(y0 + 1, min(height, round(height * 0.88)))
        projection = (binary[y0:y1] < 128).sum(axis=0).astype(np.float32)
        window = min(5, width if width % 2 else max(1, width - 1))
        if window > 1:
            projection = np.convolve(
                projection, np.ones(window, np.float32) / window, mode="same"
            )
        low = max(1, round(width * 0.30))
        high = min(width - 2, round(width * 0.70))
        if high < low:
            return width // 2
        candidates = np.arange(low, high + 1)
        scores = projection[candidates] + 0.035 * np.abs(candidates - width / 2.0)
        ink_per_column = (binary[y0:y1] < 128).sum(axis=0).astype(np.float32)
        cumulative = np.cumsum(ink_per_column)
        total_ink = float(cumulative[-1]) if len(cumulative) else 0.0
        if total_ink > 0:
            left_ink = cumulative[candidates]
            right_ink = total_ink - left_ink
            valid = ((left_ink >= total_ink * 0.15) &
                     (right_ink >= total_ink * 0.15))
            scores = np.where(valid, scores, np.inf)
            if not np.isfinite(scores).any():
                scores = projection[candidates] + 0.035 * np.abs(
                    candidates - width / 2.0
                )
        return int(candidates[int(np.argmin(scores))])

    @classmethod
    def split_binary(cls, binary: np.ndarray, split_gap: int = 3,
                     split_x: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, int]:
        """Split at the detected valley and return both digit regions."""
        width = binary.shape[1]
        if width < 2:
            return binary, binary, 0
        split_x = cls.find_split(binary) if split_x is None else int(split_x)
        gap = min(max(0, split_gap), max(0, width // 2 - 1))
        split_x = max(gap + 1, min(width - gap - 1, split_x))
        return binary[:, :split_x - gap], binary[:, split_x + gap:], split_x

    @staticmethod
    def prepare_digit_image(digit: np.ndarray) -> np.ndarray:
        """Pad one digit to square and resize to the exact CNN input pixels."""
        gray = cv2.cvtColor(digit, cv2.COLOR_BGR2GRAY) if digit.ndim == 3 else digit
        h, w = gray.shape[:2]
        side = max(h, w)
        canvas = np.full((side, side), 255, dtype=np.uint8)
        y, x = (side - h) // 2, (side - w) // 2
        canvas[y:y + h, x:x + w] = gray
        return cv2.resize(canvas, (32, 32), interpolation=cv2.INTER_AREA)

    @classmethod
    def _prepare(cls, digit: np.ndarray) -> np.ndarray:
        resized = cls.prepare_digit_image(digit)
        return (resized.astype(np.float32) / 255.0 - 0.5) / 0.5

    @torch.inference_mode()
    def predict(self, plate: np.ndarray) -> Tuple[Optional[str], float, Tuple[float, float]]:
        plate = self.binarize_plate(plate, self.split_gap)
        left, right, _split_x = self.split_binary(plate, self.split_gap)
        batch = np.stack([self._prepare(left), self._prepare(right)])[:, None]
        logits = self.model(torch.from_numpy(batch).to(self.device))
        probs = torch.softmax(logits.float(), dim=1)
        confs, digits = probs.max(dim=1)
        conf_pair = (float(confs[0]), float(confs[1]))
        if (conf_pair[0] < self.left_confidence or
                conf_pair[1] < self.right_confidence):
            return None, min(conf_pair), conf_pair
        text = f"{int(digits[0])}{int(digits[1])}"
        return text, min(conf_pair), conf_pair
