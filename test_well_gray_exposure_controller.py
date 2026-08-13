import unittest

import cv2
import numpy as np

from well_gray_exposure_controller import (
    NORMAL,
    OVEREXPOSED,
    ExposureConfig,
    ExposureController,
    classifier_gray,
    evaluate_exposure,
)


class FakeCamera:
    def __init__(self) -> None:
        self.values = []

    def set_exposure(self, value: str) -> None:
        self.values.append(value)


class GrayExposureTests(unittest.TestCase):
    def test_gray_conversion_matches_detector_conversion(self) -> None:
        image = np.asarray([[[10, 50, 200], [200, 50, 10]]], dtype=np.uint8)
        expected = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        np.testing.assert_array_equal(classifier_gray(image), expected)

    def test_threshold_boundary(self) -> None:
        normal = np.full((100, 100), 202, dtype=np.uint8)
        overexposed = np.full((100, 100), 203, dtype=np.uint8)
        self.assertEqual(evaluate_exposure(normal).state, NORMAL)
        self.assertEqual(evaluate_exposure(overexposed).state, OVEREXPOSED)

    def test_tilt_corners_do_not_change_center_decision(self) -> None:
        image = np.zeros((100, 100), dtype=np.uint8)
        image[20:80, 20:80] = 203
        result = evaluate_exposure(image)
        self.assertEqual(result.state, OVEREXPOSED)
        self.assertEqual(result.metrics["center_p10"], 203.0)

    def test_controller_requests_and_applies_decrease(self) -> None:
        config = ExposureConfig(
            initial_exposure_ns=5_000_000,
            decrease_factor=0.60,
            adjustment_cooldown_seconds=0.0,
            settle_seconds=0.0,
        )
        controller = ExposureController(config, logger=None)
        image = np.full((100, 100), 255, dtype=np.uint8)
        result = controller.observe_plate(image, exposure_generation=0)
        self.assertIsNotNone(result)
        self.assertEqual(result.state, OVEREXPOSED)
        camera = FakeCamera()
        self.assertEqual(controller.apply_pending(camera), 3_000_000)
        self.assertEqual(camera.values, ["3000000 3000000"])
        self.assertEqual(controller.snapshot(), (3_000_000, 1))

    def test_old_generation_is_ignored(self) -> None:
        config = ExposureConfig(adjustment_cooldown_seconds=0.0, settle_seconds=0.0)
        controller = ExposureController(config, logger=None)
        image = np.full((100, 100), 255, dtype=np.uint8)
        controller.observe_plate(image, exposure_generation=0)
        controller.apply_pending(FakeCamera())
        self.assertIsNone(controller.observe_plate(image, exposure_generation=0))


if __name__ == "__main__":
    unittest.main()
