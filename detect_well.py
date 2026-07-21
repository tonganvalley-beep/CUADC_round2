# python3.8 -m pip install scipy
# 压入队列的图片可以先进行一步缩放，减少队列存储压力，如果能看清的话降低
# engine
# 啧 现在的挑选逻辑是数量最多的三个里选一个数值合适的，之后要不要改成如果三个里有类别一样的就向下顺延，这得看我们识别得准还是解算得准了
# 把这个 Gst appsink 再改成 零拷贝 NVMM → CUDA TensorRT
# 改变曝光的函数 cap.set_exposure("10000000 10000000")
# save_thread 和 infer_thread 都读同一个 numpy.ndarray，不要在任何一个线程里面修改 data.img
import cv2
import numpy as np
import math
import os
import time
import torch
import socket
import signal
import yaml
import threading
import datetime

from pathlib import Path
from collections import defaultdict, deque
from ultralytics import YOLO

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst

from scipy.spatial.transform import Rotation as R

# ROS
import rospy
from sensor_msgs.msg import NavSatFix
from geometry_msgs.msg import PoseStamped
from mavros_msgs.msg import WaypointReached
from tf.transformations import euler_from_quaternion

# 参数配置

# 距离限制
dis_lim = 50
DIS_SHOW = True

# 摄像头
FPS = 5                     # 拍照帧率
exptr = "5000000 5000000"   # 曝光时长

# 读取参数
GROUND_TEST = False
SCOUT_LAT = 39.5861817
SCOUT_LON = 116.1997466
TARGET_LAT = 39.5861817
TARGET_LON = 116.1997466
SEND_WP = 9

# YOLO参数
well_imgsz = 640        # 第一轮关键点识别图片尺寸
pic_imgsz = 640         # 第二轮识别图片尺寸

well_conf = 0.6         # 第一轮关键点识别置信度
pic_conf = 0.75         # 第二轮识别置信度

QUEUE_MAX = 12
BATCH_SIZE = 2
CLS_BATCH = 6
SAVE_QUEUE_MAX=12
WELL_KPT_CONF = 0.35            # 关键点置信度阈值，低于该值的天井不进行裁剪
WELL_SIDE_MIN = 10.0            # 四边形每边的最小长度
WELL_AREA_MIN = 100.0           # 四边形面积最小值
WELL_PARALLEL_TOL_DEG = 20.0    # 四边形对边平行度容差，单位：度

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

# 网络
SEND_PORT = 10000
send_server_addr = ("127.0.0.1",SEND_PORT)
response = b''

# ROS飞机状态
plane_lat = 39.5861817
plane_lon = 116.1997466
plane_alt = 0.0
plane_pitch = 0.0
plane_yaw = 0.0
plane_roll = 0.0
current_wp = -1

# 高度初始化
alt_takeoff = 0
takeoff_set = False
alt_sum = 0
alt_count = 0

# 多线程
task_queue = deque()
queue_lock = threading.Lock()

# 图片保存队列
save_queue = deque()
save_queue_lock = threading.Lock()

# 状态锁
state_lock = threading.Lock()

running = True

# wells结果
wells = []
num_of_cut = 0
num_of_save = 0
save_dir_exp = None

class FrameData:

    def __init__(self, img, info, frame_id):

        self.img = img
        self.info = info
        self.frame_id = frame_id

# 相机类，包含初始化一些动态修改函数
class GstCamera:

    def __init__(self, pipeline):

        Gst.init(None)
        self.pipeline = Gst.parse_launch( pipeline)
        self.camera = self.pipeline.get_by_name("camera")
        self.sink = self.pipeline.get_by_name("sink")
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

    def release(self):

        self.pipeline.set_state(Gst.State.NULL)

# 信号退出
def shutdown_handler(sig, frame):

    global running
    print("[INFO] shutdown signal")
    running = False

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

def gps_cb(msg):

    global plane_lat, plane_lon

    with state_lock:
        plane_lat = msg.latitude
        plane_lon = msg.longitude

def pose_cb(msg):

    global plane_roll
    global plane_pitch
    global plane_yaw
    global plane_alt

    global alt_takeoff
    global takeoff_set
    global alt_sum
    global alt_count

    q = msg.pose.orientation
    quat=[q.x, q.y, q.z, q.w]

    alt_now=msg.pose.position.z

    with state_lock:

        plane_roll,plane_pitch,plane_yaw=euler_from_quaternion(quat)

        if not takeoff_set:
            alt_sum += alt_now
            alt_count += 1

            if alt_count>=50:
                alt_takeoff=alt_sum/alt_count
                takeoff_set=True

        else:
            plane_alt=alt_now-alt_takeoff

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
def get_well_gps(u, v, lon0, lat0, rel_alt, pitch, yaw, roll = 0.0):

    pts_dist = np.array([[[u, v]]], dtype=np.float64)

    # 修正相机畸变
    pts_undist = cv2.undistortPoints(pts_dist, K, D)

    x = pts_undist[0, 0, 0]
    y = pts_undist[0, 0, 1]

    # 相机坐标系射线
    r_cam = np.array([x, y, 1.0])
    r_cam = r_cam / np.linalg.norm(r_cam)

    # 相机 → 机体系（相机朝正下方）
    # 机体系: X前 Y右 Z下
    r_body = np.array([
        r_cam[1],  # 前
        r_cam[0],  # 右
        r_cam[2]  # 下
    ])
    r_body = r_body / np.linalg.norm(r_body)

    # 姿态旋转（机体系 → NED），yaw(Z) pitch(Y) roll(X)
    rot = R.from_euler('ZYX', [yaw, pitch, roll])
    R_mat = rot.as_matrix()

    r_ned = R_mat @ r_body

    rN, rE, rD = r_ned

    # 防止看向地平线
    if abs(rD) < 1e-6:
        print("Ray is parallel to ground, no intersection.")
        return None

    # 射线与地面交点，飞机在 (0,0,-rel_alt)
    t = rel_alt / rD

    north = t * rN
    east  = t * rE
    print(f"north, east are {north}, {east} respectively.")

    # 经纬度
    lat = lat0 + (north / EARTH_RADIUS) * RAD2DEG
    lon = lon0 + (east / (EARTH_RADIUS * np.cos(lat0 * DEG2RAD))) * RAD2DEG

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

    return _is_parallel(edges[0], edges[2], WELL_PARALLEL_TOL_DEG) and _is_parallel(edges[1], edges[3], WELL_PARALLEL_TOL_DEG)

# 天井关键点透视裁剪
def cut(img, results):

    global num_of_cut

    out = []

    if results.keypoints is None:
        return out

    kpts = results.keypoints.data.cpu().numpy()

    for kp in kpts:

        if len(kp) < 5:
            continue

        corner_conf = [float(kp[idx][2]) for idx in (1, 2, 3, 4)]
        if min(corner_conf) < WELL_KPT_CONF:
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

        pts1 = expand_quad_pixel(pts1, expand=3)
        pts2 = np.float32([[0, 0], [100, 0], [100, 100], [0, 100]])
        M = cv2.getPerspectiveTransform(pts1, pts2)
        warped = cv2.warpPerspective(img, M, (100, 100))
        crop = cv2.resize(warped, (pic_imgsz, pic_imgsz))

        if SAVE_CROPS:
            cv2.imwrite(os.path.join(folder_cut, f"{num_of_cut}.jpg"), crop)
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

# socket发送
def init_addr(sock):

    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(send_server_addr)
    sock.listen(40)

def send(lat, lon):

    global running

    # 创建 socket 转换为 Client 模式
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    
    try:
        # 连接 C++ 服务端
        sock.connect(send_server_addr)
        print(f"[INFO] Connected to GPS server. Sending data...")

        # 第一步：发送纬度
        sock.sendall(str(lat).encode())
        ack = sock.recv(100)
        if ack != b'get lat success   ':
            print(f"[ERROR] lat ack failed, got: {ack}")
            return

        # 第二步：发送经度
        sock.sendall(str(lon).encode())
        ack = sock.recv(100)
        if ack != b'get lon success   ':
            print(f"[ERROR] lon ack failed, got: {ack}")
            return

        # 第三步：发送结束标志
        sock.sendall(b'1')
        ack = sock.recv(100)
        if ack != b'send success      ':
            print(f"[ERROR] final ack failed, got: {ack}")
            return

        print("[INFO] GPS data completely sent in one connection!")

    except Exception as e:
        print(f"[ERROR] Socket communication error: {e}")
    finally:
        # 关闭客户端连接
        sock.close()

    running=False

# 选择最终目标
def choose_and_send(cls):

    if not wells:
        print("no detection but sent")
        send(TARGET_LAT, TARGET_LON)
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
        send(TARGET_LAT, TARGET_LON)
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
    send(m_lat, m_lon)
    return True

# 保存图片和飞机信息
def save_raw_and_info(img, info, frame_id):

    lon,lat,alt,pitch,yaw,roll=info

    cv2.imwrite(f"{raw_path}/{frame_id}.jpg", img)

    with open(f"{info_path}/{frame_id}.txt", "w") as f:
        f.write(
            f"{lon} {lat} {alt} "
            f"{pitch} {yaw} {roll}"
        )

# 摄像头线程
def camera_thread(cap):

    global DIS_SHOW

    frame_id = 0

    while running and not rospy.is_shutdown():

        # 获取图像
        ret, frame = cap.read()

        if not ret:
            print("[ERROR] camera failed")
            continue

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

        if (distance < dis_lim and now_alt > 10) or GROUND_TEST:

            frame_id += 1
            info = (now_lon, now_lat, now_alt, now_pitch, now_yaw, now_roll)

            data = FrameData(frame, info, frame_id)

            # 保存队列
            with save_queue_lock:
                if len(save_queue) >= SAVE_QUEUE_MAX:
                    save_queue.popleft()
                save_queue.append(data)

            # 推理队列
            with queue_lock:
                if len(task_queue) >= QUEUE_MAX:
                    task_queue.popleft()
                    print("infer too slow")
                task_queue.append(data)
        
def save_thread():

    while running or save_queue:

        data=None

        with save_queue_lock:
            if save_queue:
                data=save_queue.popleft()

        if data is None:
            time.sleep(0.01)
            continue

        save_raw_and_info(data.img,data.info,data.frame_id)

# YOLO推理线程
def infer_thread(pose, cls):

    global num_of_save

    batch_count = 0
    cls_pending_inputs = []
    cls_pending_metas = []

    def run_cls_batch(run_inputs, run_metas):
        nonlocal batch_count
        global num_of_save

        if not run_inputs:
            return

        cls_results = cls.predict(
            list(run_inputs),
            imgsz=pic_imgsz,
            conf=pic_conf,
            batch=min(CLS_BATCH, len(run_inputs)),
            device=0,
            verbose=False
        )

        for result, (wlon, wlat, crop_img) in zip(cls_results, run_metas):
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
                    cv2.imwrite(str(cls_dir / f"{num_of_save}_{idx}.jpg"), crop_img)
                    num_of_save += 1

        batch_count += 1

    while running or task_queue:

        batch=[]

        with queue_lock:
            while (task_queue and len(batch)<BATCH_SIZE):
                batch.append(task_queue.popleft())

        if not batch:
            time.sleep(0.005)
            continue

        imgs = [x.img for x in batch]
        infos = [x.info for x in batch]

        # 第一阶段 pose关键点检测
        pose_results=pose.predict(
            list(imgs),
            imgsz=well_imgsz,
            conf=well_conf,
            batch=len(imgs),
            device=0,
            verbose=False
        )

        # 地面测试
        if GROUND_TEST:
            send(TARGET_LAT, TARGET_LON)
            print("[INFO] send on the ground")
            return

        # 第二阶段 分类
        for i,res in enumerate(pose_results):
            crops=cut(imgs[i], res)
            for crop_img, (center_x, center_y) in crops:
                lon,lat,alt,pitch,yaw,roll=infos[i]
                well_gps = get_well_gps(center_x, center_y, lon, lat, alt, pitch, yaw, roll)
                if well_gps is None:
                    continue
                wlon, wlat = well_gps

                cls_pending_inputs.append(crop_img)
                cls_pending_metas.append((wlon, wlat, crop_img))

                if len(cls_pending_inputs) >= CLS_BATCH:
                    run_cls_batch(cls_pending_inputs[:CLS_BATCH], cls_pending_metas[:CLS_BATCH])
                    del cls_pending_inputs[:CLS_BATCH]
                    del cls_pending_metas[:CLS_BATCH]

    if cls_pending_inputs:
        run_cls_batch(cls_pending_inputs, cls_pending_metas)

def main():

    global GROUND_TEST
    global SCOUT_LAT
    global SCOUT_LON
    global TARGET_LAT
    global TARGET_LON
    global SEND_WP

    global running

    signal.signal(signal.SIGUSR1, shutdown_handler)

    # ROS初始化
    rospy.init_node("pic_detect_node", anonymous=True)

    rospy.Subscriber("/mavros/global_position/global", NavSatFix, gps_cb)
    rospy.Subscriber("/mavros/local_position/pose", PoseStamped, pose_cb)
    rospy.Subscriber("/mavros/mission/reached", WaypointReached, wp_cb)

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

    # 存储文件夹创建
    init_dirs()

    # 加载模型
    print("[INFO] loading YOLO...")
    det=YOLO("weights/step1_0717_960.pt")
    cls=YOLO("weights/round1_1004_tqc.pt")
    det.fuse()
    cls.fuse()
    # 模型预热
    dummy=np.zeros((well_imgsz,well_imgsz,3), dtype=np.uint8)
    det.predict(dummy, imgsz=well_imgsz, conf=well_conf, device=0, verbose=False)

    dummy=np.zeros((pic_imgsz,pic_imgsz,3), dtype=np.uint8)
    cls.predict(dummy, imgsz=pic_imgsz, conf=pic_conf, device=0, verbose=False)

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
    t_infer=threading.Thread(target=infer_thread, args=(det,cls), daemon=True)

    t_camera.start()
    t_save.start()
    t_infer.start()

    try:
        while (running and not rospy.is_shutdown() and  current_wp < SEND_WP):
            rospy.sleep(0.1)

    except KeyboardInterrupt:
        pass

    # 结束侦察，停止摄像头入队，排空剩余推理与保存任务
    running=False
    t_infer.join()

    # 发送解算结果
    print("[INFO] choosing target...")
    choose_and_send(cls)

    t_camera.join(timeout=2)
    t_save.join()

    # 释放摄像头
    if cap.pipeline is not None:
        cap.release()

    cv2.destroyAllWindows()
    print("[INFO] exit")

if __name__=="__main__":
    main()