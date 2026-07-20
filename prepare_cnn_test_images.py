"""Create 100x100 binary CNN test plates from full camera images.

Runs YOLO Pose only. It performs no ROS, GPS, socket, or shared-memory work.
"""

import argparse
import glob
import os

import cv2
import numpy as np
from ultralytics import YOLO

from digit_recognizer import DigitRecognizer


SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp")


def collect_images(items):
    result = []
    seen = set()
    for item in items:
        candidates = []
        if os.path.isdir(item):
            candidates = [os.path.join(item, name) for name in sorted(os.listdir(item))]
        else:
            candidates = glob.glob(item)
        for path in candidates:
            path = os.path.abspath(path)
            if os.path.isfile(path) and path.lower().endswith(SUFFIXES) and path not in seen:
                result.append(path)
                seen.add(path)
    return result


def read_image(path):
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)


def write_image(path, image):
    suffix = os.path.splitext(path)[1] or ".png"
    ok, encoded = cv2.imencode(suffix, image)
    if not ok:
        raise RuntimeError("encode failed: {}".format(path))
    encoded.tofile(path)


def ordered_corners(points):
    p1, p2, p3, p4 = points
    v12 = np.asarray(p2) - np.asarray(p1)
    v23 = np.asarray(p3) - np.asarray(p2)
    cross = v12[0] * v23[1] - v12[1] * v23[0]
    if cross > 0:
        return np.float32([p4, p1, p2, p3])
    return np.float32([p1, p4, p3, p2])


def expand_quad(points, pixels):
    center = points.mean(axis=0)
    result = points.copy()
    for i, point in enumerate(points):
        direction = point - center
        length = np.linalg.norm(direction)
        if length > 1e-6:
            result[i] = point + direction / length * pixels
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", help="full camera images, globs, or directories")
    parser.add_argument("--model", default="step1_0717_960.pt")
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--conf", type=float, default=0.60)
    parser.add_argument("--kpt-conf", type=float, default=0.25)
    parser.add_argument("--expand", type=float, default=3.0)
    parser.add_argument("--output", default="cnn_test_prepared")
    parser.add_argument("--device", default="0")
    args = parser.parse_args()

    paths = collect_images(args.inputs)
    if not paths:
        raise SystemExit("No supported input images found")
    for folder in ("visualized", "warp_color", "plate_binary"):
        os.makedirs(os.path.join(args.output, folder), exist_ok=True)

    model = YOLO(args.model)
    results = model.predict(paths, imgsz=args.imgsz, conf=args.conf,
                            device=args.device, verbose=False)
    total = 0
    destination = np.float32([[0, 0], [100, 0], [100, 100], [0, 100]])
    for path, result in zip(paths, results):
        stem = os.path.splitext(os.path.basename(path))[0]
        write_image(os.path.join(args.output, "visualized", stem + ".jpg"), result.plot())
        if result.keypoints is None:
            print("NO_KEYPOINTS {}".format(path))
            continue
        image = result.orig_img
        for obj_index, keypoints in enumerate(result.keypoints.data.cpu().numpy()):
            if len(keypoints) < 5 or np.min(keypoints[1:5, 2]) < args.kpt_conf:
                print("LOW_KEYPOINT {} object={}".format(path, obj_index))
                continue
            source = ordered_corners(keypoints[1:5, :2])
            source = expand_quad(source, args.expand)
            matrix = cv2.getPerspectiveTransform(source, destination)
            color = cv2.warpPerspective(image, matrix, (100, 100),
                                        borderMode=cv2.BORDER_CONSTANT,
                                        borderValue=(255, 255, 255))
            binary = DigitRecognizer.binarize_plate(color)
            name = "{}_{}".format(stem, obj_index)
            write_image(os.path.join(args.output, "warp_color", name + ".jpg"), color)
            write_image(os.path.join(args.output, "plate_binary", name + ".png"), binary)
            total += 1
            print("SAVED {} keypoint_min={:.3f}".format(name, np.min(keypoints[1:5, 2])))
    print("summary: source_images={} plates={} output={}".format(
        len(paths), total, os.path.abspath(args.output)))


if __name__ == "__main__":
    main()
