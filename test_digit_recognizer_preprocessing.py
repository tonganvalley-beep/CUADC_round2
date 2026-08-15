import unittest

import cv2
import numpy as np

from digit_recognizer import DigitRecognizer


class DigitPreprocessingTests(unittest.TestCase):
    def test_shadow_normalization_preserves_foreground_and_background(self) -> None:
        image = np.full((100, 100, 3), (35, 35, 210), dtype=np.uint8)
        image[8:92, 8:92] = (235, 235, 235)
        image[18:82, 27:35] = (30, 30, 30)
        image[18:82, 65:73] = (30, 30, 30)
        image[50:] = np.rint(image[50:].astype(np.float32) * 0.38).astype(np.uint8)

        binary = DigitRecognizer.binarize_plate(image)

        background = np.r_[binary[20:45, 42:58].ravel(),
                           binary[58:82, 42:58].ravel()]
        upper_ink = binary[22:45, 28:34]
        lower_ink = binary[58:78, 28:34]
        self.assertGreater(float(np.mean(background == 255)), 0.90)
        self.assertGreater(float(np.mean(upper_ink == 0)), 0.85)
        self.assertGreater(float(np.mean(lower_ink == 0)), 0.85)

    def test_split_follows_shifted_inter_digit_valley(self) -> None:
        binary = np.full((100, 100), 255, dtype=np.uint8)
        cv2.putText(binary, "2", (30, 78), cv2.FONT_HERSHEY_SIMPLEX,
                    2.0, 0, 6, cv2.LINE_AA)
        cv2.putText(binary, "5", (70, 78), cv2.FONT_HERSHEY_SIMPLEX,
                    2.0, 0, 6, cv2.LINE_AA)
        split_x = DigitRecognizer.find_split(binary)
        self.assertGreater(split_x, 50)
        self.assertLess(split_x, 72)


if __name__ == "__main__":
    unittest.main()
