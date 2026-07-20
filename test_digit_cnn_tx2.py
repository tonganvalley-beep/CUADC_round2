"""Standalone TX2 smoke test for rectified two-digit plate images.

This deliberately does not import ROS or Ultralytics and performs no network
or GPS operations. Inputs must already be rectified plate crops (not full
camera frames). Every input is normalized to the deployment size 100x100.
"""

import argparse
import glob
import os

import cv2
import numpy as np

from digit_recognizer import DigitRecognizer


SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp")


def collect_images(inputs):
    paths = []
    for item in inputs:
        if os.path.isdir(item):
            for name in sorted(os.listdir(item)):
                path = os.path.join(item, name)
                if os.path.isfile(path) and name.lower().endswith(SUFFIXES):
                    paths.append(path)
        else:
            matches = glob.glob(item)
            paths.extend(path for path in matches if path.lower().endswith(SUFFIXES))
    result = []
    seen = set()
    for path in paths:
        absolute = os.path.abspath(path)
        if absolute not in seen:
            result.append(absolute)
            seen.add(absolute)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", help="rectified plate files, globs, or directories")
    parser.add_argument("--model", default="weights/digit_cnn.ts")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--threshold", type=float, default=0.50)
    parser.add_argument("--gap", type=int, default=3)
    args = parser.parse_args()

    paths = collect_images(args.inputs)
    if not paths:
        raise SystemExit("No supported images found")

    # Use zero internally so the predicted digits remain visible even when
    # confidence is low; PASS/LOW is evaluated against --threshold below.
    recognizer = DigitRecognizer(args.model, device=args.device,
                                 confidence=0.0, split_gap=args.gap)
    print("model={} device={} images={}".format(
        os.path.abspath(args.model), recognizer.device, len(paths)))
    print("threshold={:.3f} gap={} runtime_plate=100x100".format(
        args.threshold, args.gap))

    passed = 0
    for index, path in enumerate(paths, 1):
        # imdecode also handles non-ASCII paths on the Windows preparation PC.
        image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            print("[{}/{}] READ_ERROR {}".format(index, len(paths), path))
            continue
        plate = cv2.resize(image, (100, 100), interpolation=cv2.INTER_LINEAR)
        text, minimum, pair = recognizer.predict(plate)
        status = "PASS" if minimum >= args.threshold else "LOW"
        if status == "PASS":
            passed += 1
        print("[{}/{}] {} pred={} left={:.4f} right={:.4f} min={:.4f} {}".format(
            index, len(paths), status, text, pair[0], pair[1], minimum,
            os.path.basename(path)))

    print("summary: pass={}/{} threshold={:.3f}".format(
        passed, len(paths), args.threshold))


if __name__ == "__main__":
    main()
