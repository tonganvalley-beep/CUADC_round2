# python3.8 -m pip install scipy
# 压入队列的图片可以先进行一步缩放，减少队列存储压力，如果能看清的话降低
# engine
# 啧 现在的挑选逻辑是数量最多的三个里选一个数值合适的，之后要不要改成如果三个里有类别一样的就向下顺延，这得看我们识别得准还是解算得准了
# 把这个 Gst appsink 再改成 零拷贝 NVMM → CUDA TensorRT
# 改变曝光的函数 cap.set_exposure("10000000 10000000")
# save_thread 和 infer_thread 都读同一个 numpy.ndarray，不要在任何一个线程里面修改 data.img
#相机持续拍摄，复原方法：注释“相机释放”那一篇注释的while循环，然后camera_running=False更改到infer.join()前面，前面不用改
import cv2
import numpy as np
import math
import os
import shutil
import time
import torch
import signal
import yaml
import threading
import datetime

from pathlib import Path
from collections import defaultdict, deque
try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

from trt_wrapper import TensorRTWrapper
try:
    # This detector converts every crop to grayscale before classification, so
    # use its dedicated grayscale-only exposure rule instead of the shared
    # color/Otsu controller.
    from well_gray_exposure_controller import ExposureConfig, ExposureController
    _exposure_import_error = None
except Exception as exc:
    # 曝光模块缺失或加载失败时，只关闭曝光功能，不影响主体检测流程。
    ExposureConfig = None
    ExposureController = None
    _exposure_import_error = exc

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst

from scipy.spatial.transform import Rotation as R

# ROS
import rospy
from sensor_msgs.msg import NavSatFix
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float64
from mavros_msgs.msg import WaypointReached
from tf.transformations import euler_from_quaternion

import sys
sys.path.insert(0, '/home/tx2/Wintter/devel/lib/python2.7/dist-packages')
from test.msg import gps as GpsMsg, yoloresults

# 参数配置

# 距离限制
dis_lim = 300
DIS_SHOW = True

# 摄像头
FPS = int(os.getenv("CAMERA_FPS", "7"))  # 实测飞行使用 7~8 fps，可覆盖为 8
_initial_exposure_ns = int(os.getenv("CAMERA_EXPOSURE_INITIAL_NS", "500000"))
exptr = f"{_initial_exposure_ns} {_initial_exposure_ns}"   # 曝光时长
ENABLE_EXPOSURE = True
ENABLE_FRAME_EXPOSURE = os.getenv("CAMERA_EXPOSURE_USE_FRAME", "0") == "1"
exposure_controller = None
if ENABLE_EXPOSURE and ExposureController is not None:
    try:
        exposure_controller = ExposureController(
            ExposureConfig.from_env(initial_exposure_ns=_initial_exposure_ns)
        )
    except Exception as exc:
        print(f"[EXPOSURE][WARN] initialization failed; disabled: {exc}")
elif ENABLE_EXPOSURE:
    print(f"[EXPOSURE][WARN] module unavailable; disabled: {_exposure_import_error}")
else:
    print("[EXPOSURE] disabled by ENABLE_EXPOSURE=0")

# 读取参数
GROUND_TEST = False
SCOUT_LAT = 39.5861817
SCOUT_LON = 116.1997466
TARGET_LAT = 39.5861817
TARGET_LON = 116.1997466
SEND_WP = 9

# YOLO参数
well_imgsz = 640        # 第一轮关键点识别图片尺寸
pic_imgsz = 224         # 第二轮识别图片尺寸

well_conf = 0.6         # 第一轮关键点识别置信度
pic_conf = 0.75         # 第二轮识别置信度
well_iou = 0.45         # 第一轮NMS IoU
trt_iou = 0.45          # TensorRT分类阶段NMS IoU

QUEUE_MAX = 12
BATCH_SIZE = 1
CLS_BATCH = 1
SAVE_QUEUE_MAX=12
WELL_KPT_CONF = 0.35            # 关键点置信度阈值，低于该值的天井不进行裁剪
WELL_SIDE_MIN = 10.0            # 四边形每边的最小长度
WELL_AREA_MIN = 100.0           # 四边形面积最小值
WELL_PARALLEL_TOL_DEG = 20.0    # 四边形对边平行度容差，单位：度
WELL_EDGE_MARGIN_MIN = 5.0              # Reject when any of keypoints 1~4 is closer than this to the frame edge.
WELL_MAX_CORNER_DEVIATION_DEG = 20.0    # Maximum allowed deviation of any quad corner from 90 degrees.
WELL_AREA_EDGE_MARGIN_PX = 20.0         # Apply the quad/bbox fill test only when keypoints 1~4 are near an edge.
WELL_MIN_QUAD_BBOX_FILL = 0.107         # Minimum quad area / detection bbox area for near-edge detections.
WELL_MAX_BLACK_BORDER_RATIO = 0.015     # Maximum near-black ratio in the outer crop border.
WELL_BLACK_BORDER_WIDTH = 5             # Border width in the 100x100 perspective crop.
WELL_BLACK_PIXEL_THRESHOLD = 8          # Near-black means every BGR channel is below this value.

# 坐标解算参数
fx = 1487.0237
fy = 1486.9391
cx = 981.5314
cy = 497.8651
k1 = -0.092534
k2 = 0.105996
p1 = 0.001179
p2 = -0.001208
k3 = 0.001222
DEG2RAD = np.pi / 180.0
RAD2DEG = 180.0 / np.pi
EARTH_RADIUS = 6378137.0  # WGS84

 # 相机内参 & 畸变
K = np.array([
    [fx, 0,  cx],
    [0,  fy, cy],
    [0,  0,  1]
], dtype=np.float64)
D = np.array([k1, k2, p1, p2, k3], dtype=np.float64)

# 原图尺寸
img_cols = 1920
img_rows = 1080

# 路径
raw_path = "/home/tx2/Wintter/raw_pic"
info_path = "/home/tx2/Wintter/info_to_ground"
folder_cut = "/home/tx2/Wintter/send_to_ground"
runs_root = Path("runs/detect_well/exp")

SAVE_CROPS = True           # 是否存储裁剪下来的天井
SAVE_WELL_SNAPS = True      # 是否存储识别过程中裁剪的部分

# 是否启用TensorRT第一阶段推理，默认开启
USE_TRT_POSE = os.getenv("DETECT_WELL_USE_TRT_POSE", "1") == "1"
# 是否启用TensorRT分类推理，默认开启
USE_TRT_CLS = os.getenv("DETECT_WELL_USE_TRT_CLS", "1") == "1"
# TRT第一阶段引擎文件路径，默认weights/step1_0717_640.engine
POSE_ENGINE_PATH = os.getenv("DETECT_WELL_POSE_ENGINE", "weights/step1_0724.engine")
# TRT分类引擎文件路径，默认weights/round1_0720.engine
CLS_ENGINE_PATH = os.getenv("DETECT_WELL_CLS_ENGINE", "weights/round1_0720.engine")
# TRT底层封装动态库路径，默认当前目录trt_wrapper下的libtrt_wrapper.so
TRT_LIB_PATH = os.getenv("DETECT_WELL_TRT_LIB", "./trt_wrapper/libtrt_wrapper.so")

# ROS publishers（替代原 socket 发送，直接发布 GPS 结果）
gps_pub = None
yoloresult_pub = None

# ROS飞机状态
plane_lat = 0.0
plane_lon = 0.0
plane_alt = 0.0
plane_pitch = 0.0
plane_yaw = 0.0
plane_roll = 0.0
current_wp = -1

# 多线程
task_queue = deque()
queue_lock = threading.Lock()

# 图片保存队列
save_queue = deque()
save_queue_lock = threading.Lock()

# 裁剪图片保存队列（异步写磁盘，避免阻塞推理线程）
crop_save_queue = deque()
crop_save_lock = threading.Lock()
CROP_SAVE_MAX = 256

# 状态锁
state_lock = threading.Lock()

other_thread_running = True
camera_running = True

# wells结果
wells = []
num_of_cut = 0
num_of_save = 0
save_dir_exp = None

class FrameData:

    def __init__(
        self, img, info, frame_id, exposure_ns,
        captured_at, exposure_generation
    ):

        self.img = img
        self.info = info
        self.frame_id = frame_id
        # Keep the capture-time exposure attached to this frame.  The save
        # thread is asynchronous and must not read a newer camera setting.
        self.exposure_ns = int(exposure_ns)
        self.captured_at = captured_at
        self.exposure_generation = exposure_generation

# 相机类，包含初始化一些动态修改函数
class GstCamera:

    def __init__(self, pipeline):

        Gst.init(None)
        self.pipeline = Gst.parse_launch( pipeline)
        self.camera = self.pipeline.get_by_name("camera")
        self.sink = self.pipeline.get_by_name("sink")
        self._exposure_read_warned = False
        self.pipeline.set_state(Gst.State.PLAYING)

    def read(self):

        sample = self.sink.emit("pull-sample")

        if sample is None:
            return False, None

        buf = sample.get_buffer()
        caps = sample.get_caps()
        width = caps.get_structure(0).get_value("width")
        height = caps.get_structure(0).get_value("height")
        data = buf.extract_dup(0,buf.get_size())
        frame = np.frombuffer(data,dtype=np.uint8)
        frame = frame.reshape(height,width,3)

        return True, frame

    def set_exposure(self, exp):

        self.camera.set_property("exposuretimerange",exp)

    def get_exposure_ns(self, fallback):
        """Return the exposure configured when the next frame is pulled."""
        try:
            value = self.camera.get_property("exposuretimerange")
            numbers = str(value).replace('"', '').split()
            if numbers:
                return int(numbers[0])
        except Exception as exc:
            if not self._exposure_read_warned:
                print(f"[EXPOSURE][WARN] cannot read camera exposure: {exc}")
                self._exposure_read_warned = True
        return int(fallback)

    def release(self):

        self.pipeline.set_state(Gst.State.NULL)

# 信号退出
def shutdown_handler(sig, frame):

    global other_thread_running
    print("[INFO] shutdown signal")
    other_thread_running = False

# 配置读取
def load_config():

    path = "/home/tx2/Wintter/src/test/config.yaml"
    with open(path,'r') as f:
        return yaml.safe_load(f)

# 初始化目录
def increment_path(path:Path, sep='_'):

    if not path.exists():
        return path

    dirs = [d for d in path.parent.glob(f"{path.stem}*")
        if d.is_dir()
    ]

    n = max(
        [
            int(d.name.replace(path.stem+sep,''))
            for d in dirs
            if sep in d.name
        ]
        +
        [0]
    )+1

    return path.parent / f"{path.stem}{sep}{n}"

def init_dirs():

    global save_dir_exp

    # 原图保存路径
    os.makedirs(raw_path, exist_ok=True)
    # 实时参数信息保存路径
    os.makedirs(info_path, exist_ok=True)
    # 回传图片保存路径
    if SAVE_CROPS:
        os.makedirs(folder_cut, exist_ok=True)
    # 识别过程截取图片保存路径
    if SAVE_WELL_SNAPS:
        save_dir_exp = increment_path(runs_root)
        save_dir_exp.mkdir(parents=True, exist_ok=True)

def clean_dirs():
    """清空 raw / info / cut 三个目录下的旧文件"""
    for d in [raw_path, info_path, folder_cut]:
        if os.path.isdir(d):
            for f in os.listdir(d):
                p = os.path.join(d, f)
                try:
                    if os.path.isfile(p) or os.path.islink(p):
                        os.unlink(p)
                    elif os.path.isdir(p):
                        shutil.rmtree(p)
                except Exception as e:
                    print(f"[WARN] clean_dirs failed for {p}: {e}")
        else:
            os.makedirs(d, exist_ok=True)
    print("[INFO] cleaned raw_path, info_path, folder_cut")

def gps_cb(msg):

    global plane_lat, plane_lon

    with state_lock:
        plane_lat = msg.latitude
        plane_lon = msg.longitude

def pose_cb(msg):

    global plane_roll
    global plane_pitch
    global plane_yaw

    q = msg.pose.orientation
    quat=[q.x, q.y, q.z, q.w]

    with state_lock:
        plane_roll,plane_pitch,plane_yaw=euler_from_quaternion(quat)

def rel_alt_cb(msg):

    global plane_alt

    with state_lock:
        plane_alt = msg.data

def wp_cb(msg):

    global current_wp

    with state_lock:
        current_wp = msg.wp_seq

# GStreamer
def gstreamer_pipeline():

    return (
        "nvarguscamerasrc name=camera "
        "sensor-id=0 "

        "exposuretimerange=\""
        + exptr +
        "\" "

        "gainrange=\"1 1\" "
        "ispdigitalgainrange=\"1 1\" "
        "aelock=true ! "

        "video/x-raw(memory:NVMM),"
        f"width={img_cols},"
        f"height={img_rows},"
        "format=NV12,"
        f"framerate={FPS}/1 ! "

        "nvvidconv ! "

        "video/x-raw,"
        "format=BGRx ! "

        "videoconvert ! "

        "video/x-raw,"
        "format=BGR ! "

        "appsink name=sink "
        "drop=true "
        "max-buffers=1 "
        "sync=false"
    )

# GPS计算
def enu_meters_per_deg(lat_deg):

    lat = math.radians(lat_deg)
    return (111412.84*math.cos(lat), 111132.92)

# 计算两个经纬度之间的距离
def dis(lon1,lat1,lon2,lat2):

    mlon,mlat = enu_meters_per_deg(lat1)
    return math.sqrt(((lon1-lon2)*mlon)**2 + ((lat1-lat2)*mlat)**2)

# 坐标解算
def get_well_gps(u, v, lon0, lat0, rel_alt, pitch, yaw, roll=0.0):
    # 相机内参与畸变参数
    K = np.array([
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)

    D = np.array([k1, k2, p1, p2, k3], dtype=np.float64)

    # 去畸变，得到归一化相机坐标。
    distorted_point = np.array([[[u, v]]], dtype=np.float64)
    undistorted_point = cv2.undistortPoints(distorted_point, K, D)
    x_cam, y_cam = undistorted_point[0, 0]

    # OpenCV相机坐标：X向右、Y向下、Z沿镜头光轴。
    ray_camera = np.array([x_cam, y_cam, 1.0], dtype=np.float64)
    ray_camera /= np.linalg.norm(ray_camera)

    # 相机坐标 -> 机体FRD坐标：X向前、Y向右、Z向下。
    # 图像上方对应机头前方，所以 forward = -camera_y。
    ray_body = np.array([
        -ray_camera[1],
         ray_camera[0],
         ray_camera[2]
    ], dtype=np.float64)
    ray_body /= np.linalg.norm(ray_body)

    # TXT姿态来自ENU约定，将其转换为NED/FRD约定。
    yaw_ned = math.pi / 2.0 - yaw
    pitch_ned = -pitch

    # 云台补偿滚转；不使用TXT中的机体roll。
    roll_ned = 0.0

    body_to_ned = R.from_euler(
        'ZYX',
        [yaw_ned, pitch_ned, roll_ned]
    ).as_matrix()

    ray_ned = body_to_ned @ ray_body
    ray_north, ray_east, ray_down = ray_ned

    # 射线必须朝向地面。
    if ray_down <= 1e-8:
        return None

    # 飞机位于地面上方rel_alt米；在NED中地面交点的Down=rel_alt。
    scale = rel_alt / ray_down
    north = scale * ray_north
    east = scale * ray_east

    # 局部东北方向距离转换为WGS84经纬度。
    lat = lat0 + (north / EARTH_RADIUS) * RAD2DEG
    lon = lon0 + (
        east / (EARTH_RADIUS * math.cos(lat0 * DEG2RAD))
    ) * RAD2DEG

    return lon, lat


def is_angle_greater_than_180_counterclockwise(p1, p2, p3):

    vector12 = np.array(p2) - np.array(p1)
    vector23 = np.array(p3) - np.array(p2)
    cross_z = vector12[0] * vector23[1] - vector12[1] * vector23[0]
    return 1 if cross_z > 0 else 0

def expand_quad_pixel(pts, expand=8):

    pts = np.array(pts, dtype=np.float32)
    result = np.zeros_like(pts)
    center = np.mean(pts, axis=0)

    for i, p in enumerate(pts):
        direction = p - center
        length = np.linalg.norm(direction)
        if length > 0:
            direction = direction / length
        result[i] = p + direction * expand

    return result

def _is_parallel(v1, v2, tol_deg):

    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return False
    sin_theta = abs(v1[0] * v2[1] - v1[1] * v2[0]) / (n1 * n2)
    return sin_theta <= math.sin(math.radians(tol_deg))

def _max_corner_deviation(pts):

    max_deviation = 0.0
    for i in range(4):
        v1 = pts[(i - 1) % 4] - pts[i]
        v2 = pts[(i + 1) % 4] - pts[i]
        denominator = np.linalg.norm(v1) * np.linalg.norm(v2)
        if denominator < 1e-6:
            return float("inf")
        cosine = np.clip(np.dot(v1, v2) / denominator, -1.0, 1.0)
        angle = math.degrees(math.acos(float(cosine)))
        max_deviation = max(max_deviation, abs(angle - 90.0))
    return max_deviation

def _black_border_ratio(crop):

    height, width = crop.shape[:2]
    border_width = min(WELL_BLACK_BORDER_WIDTH, height // 2, width // 2)
    if border_width <= 0:
        return 0.0

    border_mask = np.ones((height, width), dtype=bool)
    if height > 2 * border_width and width > 2 * border_width:
        border_mask[border_width:-border_width, border_width:-border_width] = False

    if crop.ndim == 2:
        black_pixels = crop < WELL_BLACK_PIXEL_THRESHOLD
    else:
        black_pixels = np.max(crop, axis=2) < WELL_BLACK_PIXEL_THRESHOLD
    return float(np.mean(black_pixels[border_mask]))

def is_valid_well_quad(pts):

    contour = pts.reshape(-1, 1, 2).astype(np.float32)
    if not cv2.isContourConvex(contour):
        return False

    area = abs(cv2.contourArea(contour))
    if area < WELL_AREA_MIN:
        return False

    edges = [pts[(i + 1) % 4] - pts[i] for i in range(4)]
    for edge in edges:
        if np.linalg.norm(edge) < WELL_SIDE_MIN:
            return False

    if _max_corner_deviation(pts) > WELL_MAX_CORNER_DEVIATION_DEG:
        return False

    return _is_parallel(edges[0], edges[2], WELL_PARALLEL_TOL_DEG) and _is_parallel(edges[1], edges[3], WELL_PARALLEL_TOL_DEG)

# 天井关键点透视裁剪
def cut(img, results):

    global num_of_cut

    out = []

    if results.keypoints is None:
        return out

    kpts = results.keypoints.data.cpu().numpy()
    boxes = None
    if results.boxes is not None and len(results.boxes) > 0:
        boxes = results.boxes.xyxy.cpu().numpy()
    img_h, img_w = img.shape[:2]

    for detection_index, kp in enumerate(kpts):

        if len(kp) < 5:
            continue

        corner_conf = [float(kp[idx][2]) for idx in (1, 2, 3, 4)]
        if min(corner_conf) < WELL_KPT_CONF:
            continue

        # 仅使用1~4号角点进行边缘过滤；负余量表示预测点已经落在画面外。
        corner_edge_margin = min(
            min(
                float(kp[idx][0]),
                float(kp[idx][1]),
                float(img_w - 1) - float(kp[idx][0]),
                float(img_h - 1) - float(kp[idx][1]),
            )
            for idx in (1, 2, 3, 4)
        )
        if corner_edge_margin < WELL_EDGE_MARGIN_MIN:
            continue

        p1 = [float(kp[1][0]), float(kp[1][1])]
        p2 = [float(kp[2][0]), float(kp[2][1])]
        p3 = [float(kp[3][0]), float(kp[3][1])]
        p4 = [float(kp[4][0]), float(kp[4][1])]
        center_x = (p1[0] + p2[0] + p3[0] + p4[0]) / 4.0
        center_y = (p1[1] + p2[1] + p3[1] + p4[1]) / 4.0

        f = is_angle_greater_than_180_counterclockwise(p1, p2, p3)
        if f == 0:
            pts1 = np.float32([p1, p4, p3, p2])
        else:
            pts1 = np.float32([p4, p1, p2, p3])

        if not is_valid_well_quad(pts1):
            continue

        if boxes is not None and detection_index < len(boxes):
            x1, y1, x2, y2 = [float(value) for value in boxes[detection_index][:4]]
            bbox_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            quad_area = abs(cv2.contourArea(pts1.reshape(-1, 1, 2)))
            quad_bbox_fill = quad_area / bbox_area if bbox_area > 1e-6 else 0.0
            if (corner_edge_margin < WELL_AREA_EDGE_MARGIN_PX
                    and quad_bbox_fill < WELL_MIN_QUAD_BBOX_FILL):
                continue

        pts1 = expand_quad_pixel(pts1, expand=3)
        pts2 = np.float32([[0, 0], [100, 0], [100, 100], [0, 100]])
        M = cv2.getPerspectiveTransform(pts1, pts2)
        warped = cv2.warpPerspective(img, M, (100, 100))
        crop = warped

        if _black_border_ratio(crop) > WELL_MAX_BLACK_BORDER_RATIO:
            continue

        if SAVE_CROPS:
            path = os.path.join(folder_cut, f"{num_of_cut}.jpg")
            with crop_save_lock:
                if len(crop_save_queue) >= CROP_SAVE_MAX:
                    crop_save_queue.popleft()
                crop_save_queue.append((crop, path))
            num_of_cut += 1

        out.append((crop, (center_x, center_y)))

    return out

def cut_from_boxes(img, results):

    global num_of_cut

    out = []
    if results.boxes is None or len(results.boxes) == 0:
        return out

    h, w = img.shape[:2]
    xyxy = results.boxes.xyxy
    confs = results.boxes.conf

    for i in range(len(results.boxes)):
        if confs is not None and float(confs[i]) < well_conf:
            continue

        x1, y1, x2, y2 = xyxy[i]
        x1 = int(max(0, min(w - 1, round(float(x1)))))
        y1 = int(max(0, min(h - 1, round(float(y1)))))
        x2 = int(max(0, min(w, round(float(x2)))))
        y2 = int(max(0, min(h, round(float(y2)))))
        if x2 - x1 < 8 or y2 - y1 < 8:
            continue

        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0

        if SAVE_CROPS:
            path = os.path.join(folder_cut, f"{num_of_cut}.jpg")
            with crop_save_lock:
                if len(crop_save_queue) >= CROP_SAVE_MAX:
                    crop_save_queue.popleft()
                crop_save_queue.append((crop, path))
            num_of_cut += 1

        out.append((crop, (center_x, center_y)))

    return out

# wells管理
def update_wells(new_lon, new_lat, dis_same_well=10, dis_dfrt_well=10):

    global wells

    if new_lon is None or new_lat is None:
        return None

    if not wells:
        wells.append(
            {
                "lon":new_lon,
                "lat":new_lat,
                "points":[(new_lon, new_lat)],
                "per_class":defaultdict(list)
            }
        )
        return 1

    min_dist=float('inf')
    min_idx=-1

    for i,w in enumerate(wells):
        d=dis(new_lon, new_lat, w["lon"], w["lat"])
        if d<min_dist:
            min_dist=d
            min_idx=i

    if min_dist <= dis_same_well:
        w=wells[min_idx]
        w["points"].append((new_lon, new_lat))
        lons, lats = zip(*w["points"])
        w["lon"] = sum(lons) / len(lons)
        w["lat"] = sum(lats) / len(lats)

        return min_idx+1

    elif min_dist > dis_dfrt_well:
        wells.append(
            {
                "lon":new_lon,
                "lat":new_lat,
                "points":[(new_lon, new_lat)],
                "per_class":defaultdict(list)
            }
        )

        return len(wells)
    return None

# ROS 直接发布 GPS 结果
def send_gps_via_ros(lat, lon):

    global gps_pub, yoloresult_pub, other_thread_running

    # 等待 main_ctl 接受通知
    while yoloresult_pub.get_num_connections() == 0 and not rospy.is_shutdown():
        rospy.logwarn("waiting for subscriber on 'mainctl'... current: %d", yoloresult_pub.get_num_connections())
        rospy.sleep(0.1)

    # 通知 main_ctl：YOLO 检测完成
    yolo_result = yoloresults()
    yolo_result.flag = 1
    yoloresult_pub.publish(yolo_result)

    # 等待 throw_ways 订阅 gps topic
    while gps_pub.get_num_connections() == 0 and not rospy.is_shutdown():
        rospy.logwarn("waiting for subscriber on 'gps'... current: %d", gps_pub.get_num_connections())
        rospy.sleep(0.1)

    # 发布目标 GPS
    pub_gps = GpsMsg()
    pub_gps.latitude = lat
    pub_gps.longitude = lon
    gps_pub.publish(pub_gps)
    rospy.loginfo("GPS published successfully via ROS: lat=%.7f, lon=%.7f", lat, lon)

    other_thread_running = False

# 选择最终目标
def choose_and_send_gps_via_ros(cls):

    if not wells:
        print("no detection but sent")
        send_gps_via_ros(TARGET_LAT, TARGET_LON)
        return True

    stats = []
    for idx, w in enumerate(wells):
        pts = w.get("points", [])
        if not pts:
            continue
        lons, lats = zip(*pts)
        avg_lon = sum(lons) / len(lons)
        avg_lat = sum(lats) / len(lats)
        counts = {k: len(v) for k, v in w.get("per_class", {}).items()}
        main_cls, main_cnt = (None, 0)
        if counts:
            main_cls = max(counts.items(), key=lambda x: x[1])[0]
            main_cnt = counts[main_cls]
        stats.append((idx, len(pts), avg_lon, avg_lat, main_cls, main_cnt))

    if not stats:
        print("no main_cls but sent")
        send_gps_via_ros(TARGET_LAT, TARGET_LON)
        return True

    stats = sorted(stats, key=lambda x: x[1], reverse=True)[:3]

    print("Top well groups:")
    for i, (_, _, avg_lon, avg_lat, main_cls, main_cnt) in enumerate(stats):
        cls_name = cls.names[main_cls] if main_cls is not None else "Unknown"
        print(
            f"Group {i+1}: avg_lon={avg_lon:.7f}, avg_lat={avg_lat:.7f}, "
            f"main_cls={cls_name}, main_cls_count={main_cnt}"
        )

    stats = sorted(stats, key=lambda x: x[4] if x[4] is not None else -1, reverse=True)
    _, _, m_lon, m_lat, m_cls, _ = stats[0]
    print(
        f"Send group: avg_lon={m_lon:.7f}, avg_lat={m_lat:.7f}, "
        f"main_cls={cls.names[m_cls] if m_cls is not None else 'Unknown'}"
    )
    send_gps_via_ros(m_lat, m_lon)
    return True

# 保存图片和飞机信息
def save_raw_and_info(img, info, frame_id, exposure_ns):

    lon,lat,alt,pitch,yaw,roll=info

    image_start = time.perf_counter()
    cv2.imwrite(f"{raw_path}/{frame_id}.jpg", img)
    image_elapsed = time.perf_counter() - image_start

    info_start = time.perf_counter()
    with open(f"{info_path}/{frame_id}.txt", "w") as f:
        f.write(
            f"{lon} {lat} {alt} "
            f"{pitch} {yaw} {roll} {int(exposure_ns)}"
        )
    info_elapsed = time.perf_counter() - info_start

    return image_elapsed, info_elapsed

# 摄像头线程
def camera_thread(cap):

    global DIS_SHOW

    frame_id = 0
    
    while camera_running and not rospy.is_shutdown():

        # 应用待处理曝光
        if exposure_controller is not None:
            exposure_controller.apply_pending(cap)

        # 获取图像
        ret, frame = cap.read()

        if not ret:
            print("[ERROR] camera failed")
            continue

        captured_at = time.monotonic()
        # 记录本帧对应的曝光时间与曝光代次；它们不是摄像头 ID。
        exposure_fallback = (
            exposure_controller.current_exposure_ns
            if exposure_controller is not None
            else _initial_exposure_ns
        )
        frame_exposure_ns = cap.get_exposure_ns(exposure_fallback)
        if exposure_controller is not None:
            _, exposure_generation = exposure_controller.snapshot()
        else:
            exposure_generation = None

        # 距离判断
        with state_lock:
            now_lat = plane_lat
            now_lon = plane_lon
            now_alt = plane_alt
            now_pitch = plane_pitch
            now_yaw = plane_yaw
            now_roll = plane_roll

        distance = dis(SCOUT_LON, SCOUT_LAT, now_lon, now_lat)
        if DIS_SHOW:
            print(f"dis={distance}")
            DIS_SHOW = False

        if (camera_running and (distance < dis_lim and now_alt > 10) ) or GROUND_TEST:

            if exposure_controller is not None and ENABLE_FRAME_EXPOSURE:
                exposure_controller.observe_frame(frame)

            frame_id += 1
            info = (now_lon, now_lat, now_alt, now_pitch, now_yaw, now_roll)

            data = FrameData(
                frame, info, frame_id, frame_exposure_ns,
                captured_at, exposure_generation
            )

            # 保存队列
            if (camera_running):
                with save_queue_lock:
                    if len(save_queue) >= SAVE_QUEUE_MAX:
                        save_queue.popleft()
                        print("save too slow")
                    save_queue.append(data)

            # 推理队列
            if (other_thread_running):
                with queue_lock:
                    if len(task_queue) >= QUEUE_MAX:
                        task_queue.popleft()
                        print("infer too slow")
                    task_queue.append(data)
        
def save_thread():
    
    while camera_running or other_thread_running or save_queue:

        data=None
        pending_count = 0

        with save_queue_lock:
            if save_queue:
                data=save_queue.popleft()
                pending_count = len(save_queue)

        if data is None:
            time.sleep(0.01)
            continue

        queue_wait = max(0.0, time.monotonic() - data.captured_at)
        save_start = time.perf_counter()
        image_elapsed, info_elapsed = save_raw_and_info(
            data.img, data.info, data.frame_id, data.exposure_ns
        )
        save_elapsed = time.perf_counter() - save_start
        print(
            f"[TIMING] save frame={data.frame_id} "
            f"total={save_elapsed*1000:.1f}ms "
            f"jpg={image_elapsed*1000:.1f}ms "
            f"txt={info_elapsed*1000:.1f}ms "
            f"queue_wait={queue_wait*1000:.1f}ms "
            f"pending={pending_count}"
        )

def crop_save_thread():
    """异步写裁剪图片到磁盘，避免阻塞推理线程"""
    while other_thread_running or crop_save_queue:
        item = None
        with crop_save_lock:
            if crop_save_queue:
                item = crop_save_queue.popleft()

        if item is None:
            time.sleep(0.01)
            continue

        crop_img, path = item
        try:
            cv2.imwrite(path, crop_img)
        except Exception as e:
            print(f"[ERROR] crop_save_thread imwrite failed: {e}")

# YOLO推理线程
def infer_thread(pose, cls):

    global num_of_save

    while other_thread_running or task_queue:

        data = None

        with queue_lock:
            if task_queue:
                data = task_queue.popleft()

        if data is None:
            time.sleep(0.005)
            continue

        img = data.img
        info = data.info

        # 预缩放：1920x1080 → 960x540，大幅减少 C++ 内部 CPU resize 时间
        # 关键点坐标会在推理后缩放回原始分辨率
        h0, w0 = img.shape[:2]
        det_w, det_h = 960, 540
        scale_x = float(w0) / float(det_w)
        scale_y = float(h0) / float(det_h)
        img_det = cv2.resize(img, (det_w, det_h), interpolation=cv2.INTER_NEAREST)

        t0 = time.time()

        # 第一阶段 pose关键点检测
        pose_results = pose.predict(
            img_det,
            imgsz=well_imgsz,
            conf=well_conf,
            iou=well_iou,
            batch=1,
            device=0,
            verbose=False
        )
        t1 = time.time()

        pose_result = pose_results[0] if pose_results else None
        if pose_result is None:
            continue

        # 将关键点和 bbox 坐标从检测分辨率缩放回原始分辨率
        if pose_result.keypoints is not None:
            pose_result.keypoints.data[:, :, 0] *= scale_x
            pose_result.keypoints.data[:, :, 1] *= scale_y
        if pose_result.boxes is not None and len(pose_result.boxes) > 0:
            pose_result.boxes.xyxy[:, [0, 2]] *= scale_x
            pose_result.boxes.xyxy[:, [1, 3]] *= scale_y

        if pose_result.keypoints is not None:
            crops = cut(img, pose_result)
        else:
            crops = cut_from_boxes(img, pose_result)
        if exposure_controller is not None:
            exposure_controller.observe_plates(
                (crop for crop, _ in crops),
                exposure_generation=data.exposure_generation,
                captured_at=data.captured_at,
            )
        t2 = time.time()

        t_gps_total = 0.0
        t_cls_total = 0.0
        n_wells = len(crops)

        for crop_img, (center_x, center_y) in crops:
            lon,lat,alt,pitch,yaw,roll = info
            tg0 = time.time()
            well_gps = get_well_gps(center_x, center_y, lon, lat, alt, pitch, yaw, roll)
            t_gps_total += time.time() - tg0
            if well_gps is None:
                continue
            wlon, wlat = well_gps

            tc0 = time.time()
            crop_img = cv2.cvtColor(cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
            cls_results = cls.predict(
                crop_img,
                imgsz=pic_imgsz,
                conf=pic_conf,
                iou=trt_iou,
                batch=1,
                device=0,
                verbose=False
            )
            t_cls_total += time.time() - tc0
            if not cls_results:
                continue

            result = cls_results[0]
            if result.boxes is None or len(result.boxes) == 0:
                continue

            for k in range(len(result.boxes)):
                cls_id = int(result.boxes.cls[k])
                idx = update_wells(wlon, wlat)
                if idx:
                    wells[idx - 1]["per_class"][cls_id].append((wlon, wlat))

                if SAVE_WELL_SNAPS and save_dir_exp and idx:
                    cls_dir = (save_dir_exp / f"cls_{cls_id}")
                    cls_dir.mkdir(parents=True, exist_ok=True)
                    path = str(cls_dir / f"{num_of_save}_{idx}.jpg")
                    with crop_save_lock:
                        if len(crop_save_queue) >= CROP_SAVE_MAX:
                            crop_save_queue.popleft()
                        crop_save_queue.append((crop_img, path))
                    num_of_save += 1

        t_total = time.time() - t0
        if t_total > 0.05:
            print(f"[TIMING] frame total={t_total*1000:.0f}ms "
                  f"pose={((t1-t0)*1000):.0f}ms "
                  f"cut={((t2-t1)*1000):.0f}ms "
                  f"wells={n_wells} "
                  f"gps={(t_gps_total*1000):.0f}ms "
                  f"cls={(t_cls_total*1000):.0f}ms")

def main():

    global GROUND_TEST
    global SCOUT_LAT
    global SCOUT_LON
    global TARGET_LAT
    global TARGET_LON
    global SEND_WP

    global other_thread_running,camera_running

    signal.signal(signal.SIGUSR1, shutdown_handler)

    global gps_pub, yoloresult_pub

    # ROS初始化
    rospy.init_node("pic_detect_node", anonymous=True)

    rospy.Subscriber("/mavros/global_position/global", NavSatFix, gps_cb)
    rospy.Subscriber("/mavros/local_position/pose", PoseStamped, pose_cb)
    rospy.Subscriber("/mavros/global_position/rel_alt", Float64, rel_alt_cb)
    rospy.Subscriber("/mavros/mission/reached", WaypointReached, wp_cb)

    # 直接发布检测结果到 ROS topic
    gps_pub = rospy.Publisher("gps", GpsMsg, queue_size=1)
    yoloresult_pub = rospy.Publisher("yoloresults", yoloresults, queue_size=1)

    # 配置文件
    config=load_config()

    # 地面测试状态
    GROUND_TEST=config["Ground_test"]
    # 侦察区域中心点
    SCOUT_LAT=(config["scout_target"]["lat"])
    SCOUT_LON=(config["scout_target"]["lon"])
    # 目标天井经纬度
    TARGET_LAT=(config["throw_target"]["lat"])
    TARGET_LON=(config["throw_target"]["lon"])
    # 发送结果的航点数
    SEND_WP=config["send_wp"]

    print("[INFO] ground_test:", GROUND_TEST)

    # 地面测试
    if GROUND_TEST:
        send_gps_via_ros(TARGET_LAT, TARGET_LON)
        print("[INFO] send on the ground")
        return

    # 加载模型
    print("[INFO] loading models...")
    if USE_TRT_POSE:
        print(f"[INFO] loading TensorRT pose engine: {POSE_ENGINE_PATH}")
        det = TensorRTWrapper(
            engine_path=POSE_ENGINE_PATH,
            lib_path=TRT_LIB_PATH,
            num_classes=1,
            num_keypoints=5,
            apply_sigmoid=False  # engine outputs already-decoded values
        )
    else:
        if YOLO is None:
            raise RuntimeError("ultralytics is required when TensorRT pose is disabled")
        det = YOLO("weights/step1_0724.pt")

    if USE_TRT_CLS:
        print(f"[INFO] loading TensorRT cls engine: {CLS_ENGINE_PATH}")
        cls = TensorRTWrapper(
            engine_path=CLS_ENGINE_PATH,
            lib_path=TRT_LIB_PATH,
            num_classes=12,
            num_keypoints=0,
            apply_sigmoid=False  # engine outputs already-decoded values
        )
    else:
        if YOLO is None:
            raise RuntimeError("ultralytics is required when TensorRT cls is disabled")
        cls = YOLO("weights/round1_0720.pt")

    if hasattr(det, "fuse"):
        det.fuse()
    if hasattr(cls, "fuse"):
        cls.fuse()

     # 存储文件夹创建
    init_dirs()
    
    # 清空上次运行的旧数据
    clean_dirs()

    # 模型预热
    dummy=np.zeros((well_imgsz,well_imgsz,3), dtype=np.uint8)
    det.predict(dummy, imgsz=well_imgsz, conf=well_conf, iou=well_iou, batch=1, device=0, verbose=False)

    dummy=np.zeros((pic_imgsz,pic_imgsz,3), dtype=np.uint8)
    cls.predict(dummy, imgsz=pic_imgsz, conf=pic_conf, iou=trt_iou, batch=1, device=0, verbose=False)

    # 摄像头
    pipeline=gstreamer_pipeline()
    print(pipeline)

    cap = GstCamera(pipeline)

    if cap.pipeline is None:
        print("[ERROR] camera open failed")
        return
    
    # 等待MAVROS数据
    if not GROUND_TEST:
        print("[INFO] waiting GPS...")
        while (plane_lat==0 and not rospy.is_shutdown()):
            rospy.sleep(0.1)

    # 启动线程，主程序结束后强制关闭
    t_camera=threading.Thread(target=camera_thread, args=(cap,), daemon=True)
    t_save=threading.Thread(target=save_thread, daemon=True)
    t_crop_save=threading.Thread(target=crop_save_thread, daemon=True)
    t_infer=threading.Thread(target=infer_thread, args=(det,cls), daemon=True)

    t_camera.start()
    t_save.start()
    t_crop_save.start()
    t_infer.start()

    try:
        while (other_thread_running and not rospy.is_shutdown() and  current_wp < SEND_WP):
            rospy.sleep(0.1)

    except KeyboardInterrupt:
        pass

    # 结束侦察，停止摄像头入队，排空剩余推理与保存任务
    other_thread_running=False
    t_infer.join(timeout=60)

    # 发送解算结果
    print("[INFO] choosing target...")
    choose_and_send_gps_via_ros(cls)

    #t_camera.join(timeout=2) 
    #t_save.join()
    t_crop_save.join()

    # 释放摄像头 临时添加：持续拍摄
    while(not rospy.is_shutdown()):
        if not t_camera.is_alive():  
            print("[ERROR] camera thread unexpected stoppped")
        else :
            print("camera thread is alive")
            rospy.sleep(0.5)

    camera_running=False
    t_camera.join(timeout=2)
    if cap.pipeline is not None:
        cap.release()

    t_save.join()

    cv2.destroyAllWindows()
    if hasattr(det, "close"):
        det.close()
    if hasattr(cls, "close"):
        cls.close()
    print("[INFO] exit")

if __name__=="__main__":
    main()
