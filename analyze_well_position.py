"""Analyze well keypoints, perspective crops, and image-edge proximity.

The inference and perspective-crop defaults intentionally mirror
``detect_well_exposure(4).py``: the frame is resized to 960x540 for pose
inference, coordinates are mapped back to the original frame, corner
keypoints 1..4 must have confidence >= 0.35, the quadrilateral is expanded
by 3 pixels, and the result is warped to 100x100.

The program writes reusable images/CSVs and can also open a small labeling UI
for marking each crop NORMAL, ABNORMAL, or UNCERTAIN.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
DEFAULT_INPUT = Path("unnormal_pic")
DEFAULT_MODEL = Path("step1_0724.pt")
DEFAULT_OUTPUT = Path("well_position_analysis")
DETECTION_WIDTH = 960
DETECTION_HEIGHT = 540
WELL_KPT_CONF = 0.35
WELL_SIDE_MIN = 10.0
WELL_AREA_MIN = 100.0
WELL_PARALLEL_TOL_DEG = 20.0
WINDOW_NAME = "Well perspective labeling"
LABELS = ("NORMAL", "ABNORMAL", "UNCERTAIN")


def read_image(path: Path) -> np.ndarray:
    encoded = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"cannot read image: {path}")
    return image


def write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix if path.suffix else ".jpg"
    ok, encoded = cv2.imencode(suffix, image)
    if not ok:
        raise RuntimeError(f"cannot encode image: {path}")
    encoded.tofile(str(path))


def collect_images(inputs: Iterable[Path]) -> list[Path]:
    found: set[Path] = set()
    for item in inputs:
        item = item.resolve()
        if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES:
            found.add(item)
        elif item.is_dir():
            for path in item.rglob("*"):
                if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                    found.add(path.resolve())
    return sorted(found, key=lambda p: p.as_posix().lower())


def relative_source(path: Path, roots: list[Path]) -> str:
    for root in roots:
        root = root.resolve()
        try:
            return path.relative_to(root).as_posix()
        except ValueError:
            continue
    return path.name


def is_angle_greater_than_180_counterclockwise(p1: list[float], p2: list[float], p3: list[float]) -> int:
    vector12 = np.asarray(p2) - np.asarray(p1)
    vector23 = np.asarray(p3) - np.asarray(p2)
    cross_z = vector12[0] * vector23[1] - vector12[1] * vector23[0]
    return 1 if cross_z > 0 else 0


def expand_quad_pixel(points: np.ndarray, expand: float = 3.0) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    result = np.zeros_like(points)
    center = np.mean(points, axis=0)
    for index, point in enumerate(points):
        direction = point - center
        length = np.linalg.norm(direction)
        if length > 0:
            direction /= length
        result[index] = point + direction * expand
    return result


def _is_parallel(v1: np.ndarray, v2: np.ndarray, tolerance_degrees: float) -> bool:
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return False
    sin_theta = abs(v1[0] * v2[1] - v1[1] * v2[0]) / (n1 * n2)
    return bool(sin_theta <= math.sin(math.radians(tolerance_degrees)))


def is_valid_well_quad(points: np.ndarray) -> tuple[bool, str]:
    contour = points.reshape(-1, 1, 2).astype(np.float32)
    if not cv2.isContourConvex(contour):
        return False, "non_convex_quad"
    if abs(cv2.contourArea(contour)) < WELL_AREA_MIN:
        return False, "quad_area_too_small"
    edges = [points[(index + 1) % 4] - points[index] for index in range(4)]
    if any(np.linalg.norm(edge) < WELL_SIDE_MIN for edge in edges):
        return False, "quad_side_too_short"
    if not _is_parallel(edges[0], edges[2], WELL_PARALLEL_TOL_DEG):
        return False, "opposite_sides_not_parallel"
    if not _is_parallel(edges[1], edges[3], WELL_PARALLEL_TOL_DEG):
        return False, "opposite_sides_not_parallel"
    return True, "ok"


def ordered_quad(keypoints: list[list[float]]) -> np.ndarray:
    p1 = keypoints[1][:2]
    p2 = keypoints[2][:2]
    p3 = keypoints[3][:2]
    p4 = keypoints[4][:2]
    if is_angle_greater_than_180_counterclockwise(p1, p2, p3) == 0:
        return np.float32([p1, p4, p3, p2])
    return np.float32([p4, p1, p2, p3])


def make_crop(image: np.ndarray, keypoints: list[list[float]]) -> tuple[np.ndarray | None, np.ndarray | None, str]:
    if len(keypoints) < 5:
        return None, None, "fewer_than_5_keypoints"
    corner_conf = [float(keypoints[index][2]) for index in (1, 2, 3, 4)]
    if min(corner_conf) < WELL_KPT_CONF:
        return None, None, "low_corner_confidence"
    quad = ordered_quad(keypoints)
    valid, reason = is_valid_well_quad(quad)
    if not valid:
        return None, quad, reason
    expanded = expand_quad_pixel(quad, expand=3.0)
    target = np.float32([[0, 0], [100, 0], [100, 100], [0, 100]])
    matrix = cv2.getPerspectiveTransform(expanded, target)
    return cv2.warpPerspective(image, matrix, (100, 100)), expanded, "ok"


def edge_metrics(x: float, y: float, width: int, height: int) -> dict[str, float | str]:
    distances = {
        "left": x,
        "right": (width - 1) - x,
        "top": y,
        "bottom": (height - 1) - y,
    }
    edge = min(distances, key=distances.get)
    normalized = {
        "left": x / max(width - 1, 1),
        "right": ((width - 1) - x) / max(width - 1, 1),
        "top": y / max(height - 1, 1),
        "bottom": ((height - 1) - y) / max(height - 1, 1),
    }
    margin_px = float(distances[edge])
    margin_rel = float(normalized[edge])
    return {
        # Margin is signed (negative means the prediction lies outside the
        # frame); distance is the literal non-negative distance to the edge.
        "edge_margin_px": margin_px,
        "edge_margin_rel": margin_rel,
        "edge_distance_px": abs(margin_px),
        "edge_distance_rel": abs(margin_rel),
        "nearest_edge": edge,
    }


@dataclass
class Detection:
    source: str
    frame_path: Path
    detection_index: int
    box: list[float]
    box_confidence: float
    keypoints: list[list[float]]
    crop_status: str
    crop_path: Path | None
    review_path: Path | None = None
    closest_keypoint_index: int | None = None
    closest_keypoint_x: float | None = None
    closest_keypoint_y: float | None = None
    closest_keypoint_confidence: float | None = None
    closest_edge: str = ""
    min_edge_distance_px: float | None = None
    min_edge_distance_rel: float | None = None
    min_edge_margin_px: float | None = None
    min_edge_margin_rel: float | None = None

    @property
    def key(self) -> str:
        return f"{self.source}::well_{self.detection_index:02d}"


def closest_keypoint(keypoints: list[list[float]], width: int, height: int) -> dict[str, object] | None:
    candidates: list[dict[str, object]] = []
    for index, point in enumerate(keypoints):
        if len(point) < 3 or float(point[2]) <= 0:
            continue
        x, y, confidence = map(float, point[:3])
        metrics = edge_metrics(x, y, width, height)
        candidates.append({"index": index, "x": x, "y": y, "confidence": confidence, **metrics})
    if not candidates:
        return None
    # The smallest signed margin prioritizes keypoints outside the frame,
    # which is the most useful ordering for edge-risk filtering.
    return min(candidates, key=lambda item: float(item["edge_margin_px"]))


def infer(paths: list[Path], roots: list[Path], model_path: Path, output_dir: Path,
          conf: float, iou: float, imgsz: int, device: str) -> list[Detection]:
    # Ultralytics otherwise tries to write its settings under the user profile,
    # which is unnecessary for this standalone analysis.
    # Ultralytics appends its own ``Ultralytics`` directory name to this root.
    os.environ.setdefault("YOLO_CONFIG_DIR", str(Path.cwd().resolve()))
    from ultralytics import YOLO

    model = YOLO(str(model_path.resolve()))
    detections: list[Detection] = []
    crops_dir = output_dir / "crops"
    for frame_number, path in enumerate(paths, start=1):
        image = read_image(path)
        height, width = image.shape[:2]
        resized = cv2.resize(image, (DETECTION_WIDTH, DETECTION_HEIGHT), interpolation=cv2.INTER_NEAREST)
        predict_device: str | int = device
        if device.isdigit():
            predict_device = int(device)
        results = model.predict(
            resized,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            batch=1,
            device=predict_device,
            verbose=False,
        )
        result = results[0]
        source = relative_source(path, roots)
        if result.keypoints is None:
            print(f"[{frame_number:02d}/{len(paths):02d}] {source}: no keypoints")
            continue
        keypoint_array = result.keypoints.data.cpu().numpy().copy()
        keypoint_array[:, :, 0] *= width / DETECTION_WIDTH
        keypoint_array[:, :, 1] *= height / DETECTION_HEIGHT
        if result.boxes is not None and len(result.boxes):
            boxes = result.boxes.xyxy.cpu().numpy().copy()
            boxes[:, [0, 2]] *= width / DETECTION_WIDTH
            boxes[:, [1, 3]] *= height / DETECTION_HEIGHT
            box_conf = result.boxes.conf.cpu().numpy().copy()
        else:
            boxes = np.zeros((len(keypoint_array), 4), dtype=float)
            box_conf = np.zeros((len(keypoint_array),), dtype=float)

        stem = Path(source).with_suffix("").as_posix().replace("/", "__")
        for index, points_array in enumerate(keypoint_array, start=1):
            points = [[float(value) for value in row] for row in points_array]
            crop, _expanded, status = make_crop(image, points)
            crop_path: Path | None = None
            if crop is not None:
                crop_path = crops_dir / f"{stem}__well_{index:02d}.png"
                write_image(crop_path, crop)
            closest = closest_keypoint(points, width, height)
            det = Detection(
                source=source,
                frame_path=path,
                detection_index=index,
                box=[float(value) for value in boxes[index - 1]],
                box_confidence=float(box_conf[index - 1]),
                keypoints=points,
                crop_status=status,
                crop_path=crop_path,
            )
            if closest:
                det.closest_keypoint_index = int(closest["index"])
                det.closest_keypoint_x = float(closest["x"])
                det.closest_keypoint_y = float(closest["y"])
                det.closest_keypoint_confidence = float(closest["confidence"])
                det.closest_edge = str(closest["nearest_edge"])
                det.min_edge_distance_px = float(closest["edge_distance_px"])
                det.min_edge_distance_rel = float(closest["edge_distance_rel"])
                det.min_edge_margin_px = float(closest["edge_margin_px"])
                det.min_edge_margin_rel = float(closest["edge_margin_rel"])
            detections.append(det)
        print(f"[{frame_number:02d}/{len(paths):02d}] {source}: {len(keypoint_array)} well(s)")
    return detections


def point_to_edge(point: tuple[int, int], edge: str, width: int, height: int) -> tuple[int, int]:
    x, y = point
    return {
        "left": (0, y),
        "right": (width - 1, y),
        "top": (x, 0),
        "bottom": (x, height - 1),
    }.get(edge, point)


def annotate_frame(image: np.ndarray, frame_detections: list[Detection]) -> np.ndarray:
    annotated = image.copy()
    height, width = annotated.shape[:2]
    frame_closest = min(
        (det for det in frame_detections if det.min_edge_distance_px is not None),
        key=lambda det: float(det.min_edge_margin_px),
        default=None,
    )
    for det in frame_detections:
        color = (60, 220, 60) if det.crop_status == "ok" else (0, 165, 255)
        x1, y1, x2, y2 = (int(round(value)) for value in det.box)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
        cv2.putText(annotated, f"W{det.detection_index} {det.box_confidence:.2f}",
                    (x1, max(24, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.72, color, 2, cv2.LINE_AA)
        if len(det.keypoints) >= 5:
            quad = ordered_quad(det.keypoints).round().astype(np.int32)
            cv2.polylines(annotated, [quad], True, color, 3, cv2.LINE_AA)
        for kp_index, point in enumerate(det.keypoints):
            if len(point) < 3 or point[2] <= 0:
                continue
            x, y, confidence = point[:3]
            center = (int(round(x)), int(round(y)))
            kp_color = (50, 240, 255) if confidence >= WELL_KPT_CONF else (0, 120, 255)
            cv2.circle(annotated, center, 7, (20, 20, 20), -1, cv2.LINE_AA)
            cv2.circle(annotated, center, 5, kp_color, -1, cv2.LINE_AA)
            cv2.putText(annotated, f"{kp_index}:{confidence:.2f}", (center[0] + 7, center[1] - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, kp_color, 1, cv2.LINE_AA)
    if frame_closest and frame_closest.closest_keypoint_index is not None:
        point = (int(round(frame_closest.closest_keypoint_x or 0)), int(round(frame_closest.closest_keypoint_y or 0)))
        cv2.circle(annotated, point, 13, (40, 40, 255), 3, cv2.LINE_AA)
        cv2.line(annotated, point, point_to_edge(point, frame_closest.closest_edge, width, height),
                 (40, 40, 255), 3, cv2.LINE_AA)
        text = (f"FRAME MIN: W{frame_closest.detection_index}/K{frame_closest.closest_keypoint_index} "
                f"dist={frame_closest.min_edge_distance_px:.1f}px "
                f"margin={frame_closest.min_edge_margin_px:.1f}px")
        cv2.rectangle(annotated, (8, 8), (min(width - 8, 790), 48), (15, 15, 15), -1)
        cv2.putText(annotated, text, (18, 37), cv2.FONT_HERSHEY_SIMPLEX, 0.72,
                    (60, 80, 255), 2, cv2.LINE_AA)
    return annotated


def fit_panel(image: np.ndarray, width: int, height: int, background: int = 25) -> np.ndarray:
    source_height, source_width = image.shape[:2]
    scale = min(width / max(source_width, 1), height / max(source_height, 1))
    target = (max(1, int(round(source_width * scale))), max(1, int(round(source_height * scale))))
    resized = cv2.resize(image, target, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_NEAREST)
    panel = np.full((height, width, 3), background, dtype=np.uint8)
    x = (width - resized.shape[1]) // 2
    y = (height - resized.shape[0]) // 2
    panel[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
    return panel


def make_review_panel(annotated: np.ndarray, crop: np.ndarray | None, det: Detection,
                      label: str = "UNLABELED") -> np.ndarray:
    canvas = np.full((900, 1600, 3), 18, dtype=np.uint8)
    canvas[70:850, 20:1120] = fit_panel(annotated, 1100, 780)
    if crop is not None:
        canvas[160:580, 1160:1560] = fit_panel(crop, 400, 420)
    else:
        cv2.rectangle(canvas, (1160, 160), (1560, 580), (55, 55, 55), -1)
        cv2.putText(canvas, "NO CROP", (1245, 385), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (180, 180, 180), 3)
    cv2.putText(canvas, f"{det.source}  WELL {det.detection_index}", (22, 44),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (240, 240, 240), 2, cv2.LINE_AA)
    cv2.putText(canvas, "FULL FRAME: bbox + quad + keypoints", (35, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (80, 235, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, "100x100 PERSPECTIVE CROP", (1170, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, (80, 235, 255), 2, cv2.LINE_AA)
    info = [
        f"crop_status: {det.crop_status}",
        f"closest: K{det.closest_keypoint_index} -> {det.closest_edge}",
        f"distance: {det.min_edge_distance_px:.2f}px" if det.min_edge_distance_px is not None else "distance: n/a",
        f"margin: {det.min_edge_margin_px:.2f}px ({det.min_edge_margin_rel:.6f})" if det.min_edge_margin_px is not None else "margin: n/a",
        f"label: {label}",
    ]
    colors = [(220, 220, 220)] * 4 + [
        (70, 225, 90) if label == "NORMAL" else (70, 90, 255) if label == "ABNORMAL" else (80, 210, 235)
    ]
    for row, (value, color) in enumerate(zip(info, colors)):
        cv2.putText(canvas, value, (1170, 635 + row * 42), cv2.FONT_HERSHEY_SIMPLEX,
                    0.62, color, 2 if row == 4 else 1, cv2.LINE_AA)
    return canvas


def save_outputs(paths: list[Path], detections: list[Detection], output_dir: Path,
                 labels: dict[str, str]) -> None:
    annotated_dir = output_dir / "annotated_frames"
    review_dir = output_dir / "review_panels"
    by_path: dict[Path, list[Detection]] = {path: [] for path in paths}
    for det in detections:
        by_path.setdefault(det.frame_path, []).append(det)
    annotated_cache: dict[Path, np.ndarray] = {}
    for path in paths:
        image = read_image(path)
        annotated = annotate_frame(image, by_path.get(path, []))
        annotated_cache[path] = annotated
        write_image(annotated_dir / f"{path.stem}__annotated.jpg", annotated)
    for det in detections:
        crop = read_image(det.crop_path) if det.crop_path else None
        panel = make_review_panel(annotated_cache[det.frame_path], crop, det, labels.get(det.key, "UNLABELED"))
        det.review_path = review_dir / f"{Path(det.source).stem}__well_{det.detection_index:02d}.jpg"
        write_image(det.review_path, panel)
    save_contact_sheets(detections, output_dir)
    write_csvs(paths, detections, output_dir, labels)
    write_manifest(detections, output_dir)


def save_contact_sheets(detections: list[Detection], output_dir: Path) -> None:
    """Save 2x2 review sheets for fast visual audit without opening the UI."""

    sheet_dir = output_dir / "contact_sheets"
    for offset in range(0, len(detections), 4):
        sheet = np.full((900, 1600, 3), 18, dtype=np.uint8)
        for slot, det in enumerate(detections[offset:offset + 4]):
            if not det.review_path:
                continue
            panel = read_image(det.review_path)
            thumbnail = cv2.resize(panel, (800, 450), interpolation=cv2.INTER_AREA)
            x = (slot % 2) * 800
            y = (slot // 2) * 450
            sheet[y:y + 450, x:x + 800] = thumbnail
            cv2.rectangle(sheet, (x, y), (x + 799, y + 449), (210, 210, 210), 1)
        write_image(sheet_dir / f"review_{offset + 1:03d}_{min(offset + 4, len(detections)):03d}.jpg", sheet)


def fmt(value: float | None, digits: int = 6) -> str:
    return "" if value is None else f"{value:.{digits}f}"


def write_csvs(paths: list[Path], detections: list[Detection], output_dir: Path,
               labels: dict[str, str]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    detection_fields = [
        "record_key", "source", "detection_index", "image_width", "image_height",
        "box_x1", "box_y1", "box_x2", "box_y2", "box_confidence", "crop_status",
        "perspective_label", "closest_keypoint_index", "closest_keypoint_x", "closest_keypoint_y",
        "closest_keypoint_confidence", "closest_edge", "min_edge_distance_px", "min_edge_distance_rel",
        "min_edge_margin_px", "min_edge_margin_rel",
        "crop_path", "review_path",
    ] + [f"kpt{k}_{field}" for k in range(5) for field in (
        "x", "y", "confidence", "x_rel", "y_rel", "edge_px", "edge_rel",
        "edge_margin_px", "edge_margin_rel", "nearest_edge"
    )]
    with (output_dir / "well_detections.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=detection_fields)
        writer.writeheader()
        for det in detections:
            height, width = read_image(det.frame_path).shape[:2]
            row: dict[str, object] = {
                "record_key": det.key, "source": det.source, "detection_index": det.detection_index,
                "image_width": width, "image_height": height,
                "box_x1": fmt(det.box[0], 3), "box_y1": fmt(det.box[1], 3),
                "box_x2": fmt(det.box[2], 3), "box_y2": fmt(det.box[3], 3),
                "box_confidence": fmt(det.box_confidence), "crop_status": det.crop_status,
                "perspective_label": labels.get(det.key, ""),
                "closest_keypoint_index": det.closest_keypoint_index,
                "closest_keypoint_x": fmt(det.closest_keypoint_x, 3),
                "closest_keypoint_y": fmt(det.closest_keypoint_y, 3),
                "closest_keypoint_confidence": fmt(det.closest_keypoint_confidence),
                "closest_edge": det.closest_edge,
                "min_edge_distance_px": fmt(det.min_edge_distance_px, 3),
                "min_edge_distance_rel": fmt(det.min_edge_distance_rel),
                "min_edge_margin_px": fmt(det.min_edge_margin_px, 3),
                "min_edge_margin_rel": fmt(det.min_edge_margin_rel),
                "crop_path": det.crop_path.as_posix() if det.crop_path else "",
                "review_path": det.review_path.as_posix() if det.review_path else "",
            }
            for k in range(5):
                if k >= len(det.keypoints):
                    continue
                x, y, confidence = det.keypoints[k][:3]
                metrics = edge_metrics(x, y, width, height)
                row.update({
                    f"kpt{k}_x": fmt(x, 3), f"kpt{k}_y": fmt(y, 3),
                    f"kpt{k}_confidence": fmt(confidence),
                    f"kpt{k}_x_rel": fmt(x / max(width - 1, 1)),
                    f"kpt{k}_y_rel": fmt(y / max(height - 1, 1)),
                    f"kpt{k}_edge_px": fmt(float(metrics["edge_distance_px"]), 3),
                    f"kpt{k}_edge_rel": fmt(float(metrics["edge_distance_rel"])),
                    f"kpt{k}_edge_margin_px": fmt(float(metrics["edge_margin_px"]), 3),
                    f"kpt{k}_edge_margin_rel": fmt(float(metrics["edge_margin_rel"])),
                    f"kpt{k}_nearest_edge": metrics["nearest_edge"],
                })
            writer.writerow(row)

    by_path: dict[Path, list[Detection]] = {path: [] for path in paths}
    for det in detections:
        by_path.setdefault(det.frame_path, []).append(det)
    frame_fields = ["source", "image_width", "image_height", "detection_count", "valid_crop_count",
                    "labeled_count", "frame_closest_detection_index", "frame_closest_keypoint_index",
                    "frame_closest_x", "frame_closest_y", "frame_closest_confidence", "frame_closest_edge",
                    "frame_min_edge_distance_px", "frame_min_edge_distance_rel",
                    "frame_min_edge_margin_px", "frame_min_edge_margin_rel"]
    with (output_dir / "frame_summary.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=frame_fields)
        writer.writeheader()
        for path in paths:
            frame_dets = by_path.get(path, [])
            height, width = read_image(path).shape[:2]
            closest = min((d for d in frame_dets if d.min_edge_margin_px is not None),
                          key=lambda d: float(d.min_edge_margin_px), default=None)
            row = {
                "source": frame_dets[0].source if frame_dets else path.name,
                "image_width": width, "image_height": height,
                "detection_count": len(frame_dets),
                "valid_crop_count": sum(d.crop_status == "ok" for d in frame_dets),
                "labeled_count": sum(bool(labels.get(d.key)) for d in frame_dets),
                "frame_closest_detection_index": closest.detection_index if closest else "",
                "frame_closest_keypoint_index": closest.closest_keypoint_index if closest else "",
                "frame_closest_x": fmt(closest.closest_keypoint_x, 3) if closest else "",
                "frame_closest_y": fmt(closest.closest_keypoint_y, 3) if closest else "",
                "frame_closest_confidence": fmt(closest.closest_keypoint_confidence) if closest else "",
                "frame_closest_edge": closest.closest_edge if closest else "",
                "frame_min_edge_distance_px": fmt(closest.min_edge_distance_px, 3) if closest else "",
                "frame_min_edge_distance_rel": fmt(closest.min_edge_distance_rel) if closest else "",
                "frame_min_edge_margin_px": fmt(closest.min_edge_margin_px, 3) if closest else "",
                "frame_min_edge_margin_rel": fmt(closest.min_edge_margin_rel) if closest else "",
            }
            writer.writerow(row)


def write_manifest(detections: list[Detection], output_dir: Path) -> None:
    records = []
    for det in detections:
        record = det.__dict__.copy()
        record["frame_path"] = str(det.frame_path)
        record["crop_path"] = str(det.crop_path) if det.crop_path else None
        record["review_path"] = str(det.review_path) if det.review_path else None
        records.append(record)
    (output_dir / "detections.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")


def load_labels(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return {row["record_key"]: row["perspective_label"] for row in csv.DictReader(handle)
                if row.get("record_key") and row.get("perspective_label") in LABELS}


def save_labels(path: Path, detections: list[Detection], labels: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=("record_key", "source", "detection_index", "perspective_label"))
        writer.writeheader()
        for det in detections:
            writer.writerow({"record_key": det.key, "source": det.source,
                             "detection_index": det.detection_index,
                             "perspective_label": labels.get(det.key, "")})
    os.replace(temporary, path)


class LabelingUI:
    def __init__(self, detections: list[Detection], labels: dict[str, str], labels_path: Path,
                 output_dir: Path, all_paths: list[Path]) -> None:
        if not detections:
            raise SystemExit("no wells detected")
        self.detections = detections
        self.labels = labels
        self.labels_path = labels_path
        self.output_dir = output_dir
        self.all_paths = all_paths
        self.index = next((i for i, d in enumerate(detections) if not labels.get(d.key)), 0)
        self.running = True
        self.undo_stack: list[tuple[str, str]] = []

    def render(self) -> np.ndarray:
        det = self.detections[self.index]
        panel = read_image(det.review_path) if det.review_path else np.zeros((900, 1600, 3), np.uint8)
        label = self.labels.get(det.key, "UNLABELED")
        panel = make_review_panel(annotate_frame(read_image(det.frame_path),
                                  [d for d in self.detections if d.frame_path == det.frame_path]),
                                  read_image(det.crop_path) if det.crop_path else None, det, label)
        cv2.rectangle(panel, (0, 850), (1600, 899), (35, 35, 35), -1)
        text = (f"{self.index + 1}/{len(self.detections)}  "
                "1=NORMAL  0=ABNORMAL  U=UNCERTAIN  A/D=prev/next  Z=undo  Q=save+exit")
        cv2.putText(panel, text, (28, 883), cv2.FONT_HERSHEY_SIMPLEX, 0.64,
                    (245, 245, 245), 2, cv2.LINE_AA)
        return panel

    def set_label(self, label: str) -> None:
        det = self.detections[self.index]
        self.undo_stack.append((det.key, self.labels.get(det.key, "")))
        self.labels[det.key] = label
        self.persist()
        for offset in range(1, len(self.detections) + 1):
            candidate = (self.index + offset) % len(self.detections)
            if not self.labels.get(self.detections[candidate].key):
                self.index = candidate
                return
        self.index = min(self.index + 1, len(self.detections) - 1)

    def persist(self) -> None:
        save_labels(self.labels_path, self.detections, self.labels)
        write_csvs(self.all_paths, self.detections, self.output_dir, self.labels)

    def run(self) -> None:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, 1400, 800)
        try:
            while self.running:
                cv2.imshow(WINDOW_NAME, self.render())
                raw = cv2.waitKeyEx(50)
                if raw == -1:
                    try:
                        if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                            break
                    except cv2.error:
                        break
                    continue
                key = raw & 0xFF
                if key == ord("1"):
                    self.set_label("NORMAL")
                elif key == ord("0"):
                    self.set_label("ABNORMAL")
                elif key in (ord("u"), ord("U")):
                    self.set_label("UNCERTAIN")
                elif key in (ord("a"), ord("A")) or raw in (2424832, 65361):
                    self.index = max(0, self.index - 1)
                elif key in (ord("d"), ord("D")) or raw in (2555904, 65363):
                    self.index = min(len(self.detections) - 1, self.index + 1)
                elif key in (ord("z"), ord("Z"), 8) and self.undo_stack:
                    record_key, previous = self.undo_stack.pop()
                    if previous:
                        self.labels[record_key] = previous
                    else:
                        self.labels.pop(record_key, None)
                    self.index = next(i for i, d in enumerate(self.detections) if d.key == record_key)
                    self.persist()
                elif key in (ord("q"), ord("Q"), 27):
                    break
        finally:
            self.persist()
            cv2.destroyAllWindows()
            # Keep the saved review panels/contact sheets synchronized with
            # labels changed during this UI session.
            save_outputs(self.all_paths, self.detections, self.output_dir, self.labels)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="*", type=Path, default=[DEFAULT_INPUT])
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--conf", type=float, default=0.60)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0", help="Ultralytics device, e.g. 0 or cpu")
    parser.add_argument("--no-gui", action="store_true", help="only generate images and CSV files")
    parser.add_argument("--check", action="store_true", help="validate paths without running inference")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    roots = [path.resolve() for path in args.inputs]
    paths = collect_images(args.inputs)
    if not paths:
        raise SystemExit("no input images found")
    if not args.model.is_file():
        raise SystemExit(f"model not found: {args.model}")
    if args.check:
        print(f"images: {len(paths)}")
        print(f"model: {args.model.resolve()}")
        print(f"output: {args.output_dir.resolve()}")
        return
    output_dir = args.output_dir.resolve()
    labels_path = output_dir / "perspective_labels.csv"
    labels = load_labels(labels_path)
    detections = infer(paths, roots, args.model, output_dir, args.conf, args.iou, args.imgsz, args.device)
    save_outputs(paths, detections, output_dir, labels)
    save_labels(labels_path, detections, labels)
    print(f"detections: {len(detections)}")
    print(f"results: {output_dir}")
    if not args.no_gui:
        LabelingUI(detections, labels, labels_path, output_dir, paths).run()


if __name__ == "__main__":
    main()
