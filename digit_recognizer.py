"""TorchScript two-digit recognizer used by detect_num_gps.py."""

from __future__ import annotations

import cv2
import numpy as np
import torch
from typing import Optional, Tuple


class DigitRecognizer:
    def __init__(self, model_path: str, device: str = "cuda", confidence: float = 0.80,
                 split_gap: int = 3):
        self.device = torch.device(device if device == "cpu" or torch.cuda.is_available() else "cpu")
        self.model = torch.jit.load(model_path, map_location=self.device).eval()
        self.confidence = confidence
        self.split_gap = max(0, split_gap)

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
        if min(conf_pair) < self.confidence:
            return None, min(conf_pair), conf_pair
        text = f"{int(digits[0])}{int(digits[1])}"
        return text, min(conf_pair), conf_pair
