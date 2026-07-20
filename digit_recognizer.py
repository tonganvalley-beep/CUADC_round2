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
    def binarize_plate(plate: np.ndarray) -> np.ndarray:
        """Match the binary preprocessing used to build the CNN dataset."""
        gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY) if plate.ndim == 3 else plate
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        _, binary = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        return binary

    @staticmethod
    def _prepare(digit: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(digit, cv2.COLOR_BGR2GRAY) if digit.ndim == 3 else digit
        h, w = gray.shape[:2]
        side = max(h, w)
        canvas = np.full((side, side), 255, dtype=np.uint8)
        y, x = (side - h) // 2, (side - w) // 2
        canvas[y:y + h, x:x + w] = gray
        resized = cv2.resize(canvas, (32, 32), interpolation=cv2.INTER_AREA)
        return (resized.astype(np.float32) / 255.0 - 0.5) / 0.5

    @torch.inference_mode()
    def predict(self, plate: np.ndarray) -> Tuple[Optional[str], float, Tuple[float, float]]:
        plate = self.binarize_plate(plate)
        width = plate.shape[1]
        middle = width // 2
        gap = min(self.split_gap, max(0, middle - 1))
        left = plate[:, :max(1, middle - gap)]
        right = plate[:, min(width - 1, middle + gap):]
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
