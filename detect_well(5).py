import cv2
import numpy as np
import math
from ultralytics import YOLO
from collections import defaultdict, deque
import time
import os
import torch
import socket
import struct
import threading
import datetime
from pathlib import Path
import mmap
import yaml
import signal
from scipy.spatial.transform import Rotation as R

#需要修改的变量
well_imgsz = 320        #识别天井时的图片大小
pic_imgsz = 640         #识别图像时的图片大小
well_conf = 0.5         #天井识别置信度
pic_conf = 0.75         #图像识别置信度
circle_wp = 9

# ---------- 坐标解算常量 ----------
DEG2RAD = math.pi / 180.0
RAD2DEG = 180.0 / math.pi
EARTH_RADIUS = 6378137.0  # WGS84 长半轴 (m)

# ---------- 相机内参 (需标定!) ----------
cam_fx = 800.0     # TODO: 替换为实际标定值
cam_fy = 800.0
cam_cx = 960.0     # 主点 x (img_cols/2)
cam_cy = 540.0     # 主点 y (img_rows/2)
cam_k1 = 0.0       # 径向畸变
cam_k2 = 0.0
cam_k3 = 0.0
cam_p1 = 0.0       # 切向畸变
cam_p2 = 0.0

raw_path =  "/home/duidi/Wintter/raw_pic"            #原图存储位置
info_path = "/home/duidi/Wintter/info_to_ground"     #信息存储位置
SAVE_CROPS = True       #是否保存识别到的天井
folder_cut = "/home/duidi/Wintter/send_to_ground"

SAVE_WELL_SNAPS = True  #是否保存识别到的图像
runs_root = Path("runs/detect_well/exp")

BATCH_SIZE = 6          #一批次处理的图片数
QUEUE_MAX = 12          #线程队列保存的最大数量

img_cols = 1920         #原图宽
img_rows = 1080         #原图高

SEND_PORT = 10000       #最终结果发送
send_server_addr = ('127.0.0.1', SEND_PORT)
response = ''

#共享内存
HEADER_FMT = '=iii6d'                               #来自c++的信息格式，int... double...
HEADER_SIZE = struct.calcsize(HEADER_FMT)           #按照STRUCT_FORMAT计算结构体大小
TOTAL_SIZE = HEADER_SIZE + img_cols * img_rows * 3  #加上图片

def load_config():
    with open('/home/duidi/Wintter/src/test/config.yaml', 'r') as f:
        return yaml.safe_load(f)

def open_shm():
    while True:
        try:
            fd = os.open("/dev/shm/my_shm", os.O_RDWR)
            mm = mmap.mmap(fd, TOTAL_SIZE)
            print("[INFO] shm connected")
            return mm
        except FileNotFoundError:
            print("[WAIT] waiting for C++ shm...")
            time.sleep(0.5)

mm = open_shm()

#全局
task_queue = deque()
queue_lock = threading.Lock()

wells = []
running = True
num_of_cut = 0
num_of_save = 0
save_dir_exp = None

#初始化目录
def increment_path(path: Path, sep='_'):
    if not path.exists():
        return path
    dirs = [d for d in path.parent.glob(f"{path.stem}*") if d.is_dir()]
    n = max([int(d.name.replace(path.stem + sep, '')) for d in dirs if sep in d.name] + [0]) + 1
    return path.parent / f"{path.stem}{sep}{n}"

def init_dirs():
    global save_dir_exp
    if SAVE_CROPS:
        os.makedirs(folder_cut, exist_ok=True)
    if SAVE_WELL_SNAPS:
        save_dir_exp = increment_path(runs_root)
        save_dir_exp.mkdir(parents=True, exist_ok=True)


#计算当前纬度每度对应的米数
def enu_meters_per_deg(lat_deg):
    lat = math.radians(lat_deg)
    return 111412.84 * math.cos(lat), 111132.92

#计算两点间距离
def dis(lon1, lat1, lon2, lat2):
    m_per_deg_lon, m_per_deg_lat = enu_meters_per_deg(lat1)
    return math.sqrt(((lon1 - lon2) * m_per_deg_lon) ** 2 + ((lat1 - lat2) * m_per_deg_lat) ** 2)

#坐标解算 (基于相机内参 + 畸变修正)
def pixel_to_gps(u, v, lon0, lat0, rel_alt, pitch_deg, yaw_deg, roll_deg=0.0):
    pitch = pitch_deg * DEG2RAD
    yaw   = yaw_deg * DEG2RAD
    roll  = roll_deg * DEG2RAD

    # 相机内参 & 畸变
    K = np.array([
        [cam_fx, 0,      cam_cx],
        [0,      cam_fy, cam_cy],
        [0,      0,      1]
    ], dtype=np.float64)

    D = np.array([cam_k1, cam_k2, cam_p1, cam_p2, cam_k3], dtype=np.float64)

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
        r_cam[2]   # 下
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

    # 经纬度
    lat = lat0 + (north / EARTH_RADIUS) * RAD2DEG
    lon = lon0 + (east / (EARTH_RADIUS * np.cos(lat0 * DEG2RAD))) * RAD2DEG

    return lon, lat

#cut
def cut(img, results, pad=15):
    global num_of_cut
    out = []

    if results.boxes is None:
        return out

    for box in results.boxes.xyxy:
        x1, y1, x2, y2 = map(int, box)

        cx = (x1 + x2)//2
        cy = (y1 + y2)//2
        side = max(x2-x1, y2-y1) + 2*pad

        nx1 = max(cx - side//2, 0)
        ny1 = max(cy - side//2, 0)
        nx2 = min(nx1 + side, img.shape[1])
        ny2 = min(ny1 + side, img.shape[0])

        crop = img[ny1:ny2, nx1:nx2]
        crop = cv2.resize(crop, (pic_imgsz, pic_imgsz))

        if SAVE_CROPS:
            out_name = os.path.join(folder_cut, f"{num_of_cut}.jpg")
            cv2.imwrite(out_name, crop)
            num_of_cut += 1

        out.append((crop, (nx1, ny1, side)))

    return out

#wells更新
def update_wells(new_lon, new_lat, cls_id):
    global wells

    if not wells:
        wells.append({
            "lon": new_lon,
            "lat": new_lat,
            "points": [(new_lon, new_lat)],
            "per_class": defaultdict(list)
        })
        wells[0]["per_class"][cls_id].append((new_lon, new_lat))
        return 1

    min_dist = float('inf')
    min_idx = -1

    for i, w in enumerate(wells):
        d = dis(new_lon, new_lat, w["lon"], w["lat"])
        if d < min_dist:
            min_dist = d
            min_idx = i

    if min_dist < 10:
        w = wells[min_idx]
        w["points"].append((new_lon, new_lat))
        w["per_class"][cls_id].append((new_lon, new_lat))
        return min_idx + 1
    else:
        wells.append({
            "lon": new_lon,
            "lat": new_lat,
            "points": [(new_lon, new_lat)],
            "per_class": defaultdict(list)
        })
        wells[-1]["per_class"][cls_id].append((new_lon, new_lat))
        return len(wells)

#socket发送
def init_addr(sockfd):
    sockfd.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sockfd.bind(send_server_addr)
    sockfd.listen(40)

def send(lat, lon):
    global response

    while response == '':
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        init_addr(sock)
        conn, _ = sock.accept()
        conn.sendall(str(lat).encode())
        response = conn.recv(100)
        conn.close()

    while response == b'get lat success   ':
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        init_addr(sock)
        conn, _ = sock.accept()
        conn.sendall(str(lon).encode())
        response = conn.recv(100)
        conn.close()

    while response == b'get lon success   ':
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        init_addr(sock)
        conn, _ = sock.accept()
        conn.sendall('1'.encode())
        response = conn.recv(100)
        conn.close()

    pid_str = os.popen("pgrep -n -f save_pic").read().strip()
    
    if pid_str and len(pid_str) > 0:
        pid = int(pid_str)
        print(f"Closing node PID: {pid}")
        os.kill(pid, signal.SIGUSR1)
    else:
        print("Node already closed, nothing to do.")

#结果的确定和发送
def choose_and_send(cls):
    if not wells:
        print("no detection but sent")
        send(lat, lon)
        return True
    stats = []
    for idx, w in enumerate(wells):
        pts = w.get('points', [])
        if not pts:
            continue
        lons, lats = zip(*pts)
        avg_lon = sum(lons)/len(lons); avg_lat = sum(lats)/len(lats)
        counts = {k: len(v) for k, v in w.get('per_class', {}).items()}
        main_cls, main_cnt = (None, 0)
        if counts:
            main_cls = max(counts.items(), key=lambda x: x[1])[0]
            main_cnt = counts[main_cls]
        stats.append((idx, len(pts), avg_lon, avg_lat, main_cls, main_cnt))
    if not stats:
        print("no main_cls but sent")
        send(lat, lon)
        return True
    stats = sorted(stats, key=lambda x: x[1], reverse=True)[:10]

    # 输出三组信息
    print("Top well groups:")
    for i, (idx, length, avg_lon, avg_lat, main_cls, main_cnt) in enumerate(stats):
        cls_name = cls.names[main_cls] if main_cls is not None else "Unknown"
        print(f"Group {i+1}: avg_lon={avg_lon:.7f}, avg_lat={avg_lat:.7f}, main_cls={cls_name}, main_cls_count={main_cnt}")

    stats = sorted(stats, key=lambda x: x[4], reverse=True)
    best_idx, _, m_lon, m_lat, m_cls, m_cnt = stats[0]
    print(f"Send group: avg_lon={m_lon:.7f}, avg_lat={m_lat:.7f}, main_cls={cls.names[m_cls] if m_cls is not None else 'Unknown'}")

    send(m_lat, m_lon)
    return True

#原图和信息保存
def save_raw_and_info(img, info, frame_id):
    global raw_path, info_path
    lon, lat, alt, pitch, yaw, roll = info

    cv2.imwrite(f"{raw_path}/{frame_id}.jpg", img)

    with open(f"{info_path}/{frame_id}.txt", "w") as f:
        f.write(f"{lon} {lat} {alt} {pitch} {yaw} {roll}")

num_of_reach = 0
#共享内存线程
def shm_thread():
    global running, mm, num_of_reach, circle_wp

    while running:
        try:
            flag = struct.unpack('i', mm[:4])[0]
        except:
            print("[ERROR] shm lost, reconnecting...")
            mm = open_shm()
            continue

        idle_count = 0

        if flag != 2:
            idle_count += 1
            if idle_count < 50:
                time.sleep(0.001)
            else:
                time.sleep(0.005)
            continue

        idle_count = 0

        data = struct.unpack(HEADER_FMT, mm[:HEADER_SIZE])
        frame_id, wp = data[1], data[2]   
 
        if wp == circle_wp:
            send(lat_target,lon_target)  # zhi hou gai cheng choose and send
            print("[INFO] mission complete")
            running = False
            return

        lon, lat, alt, pitch, yaw, roll = data[3:]

        img = np.frombuffer(mm[HEADER_SIZE:], dtype=np.uint8)\
                .reshape((img_rows, img_cols, 3))

        save_raw_and_info(
            img,
            (lon, lat, alt, pitch, yaw, roll),
            frame_id
        )

        with queue_lock:
            if len(task_queue) >= QUEUE_MAX:
                task_queue.popleft()
            task_queue.append((img.copy(), (lon, lat, alt, pitch, yaw, roll)))

        mm[:4] = struct.pack('i', 0)

#推理线程
def infer_thread(det, cls):
    global num_of_save, running

    while running or task_queue:

        batch = []
        with queue_lock:
            while task_queue and len(batch) < BATCH_SIZE:
                batch.append(task_queue.popleft())

        if not batch:
            time.sleep(0.005)
            continue

        imgs, infos = zip(*batch)

        det_results = det.predict(list(imgs),
                                 imgsz=well_imgsz,
                                 conf=well_conf,
                                 batch=len(imgs),
                                 device=0,
                                 verbose=False)
        #print("detecting")
        if ground_test:
            send(lat_target,lon_target)
            print("sent on the ground")
            running = False
            return

        for i, res in enumerate(det_results):

            crops = cut(imgs[i], res)

            for crop_img, (rx, ry, side) in crops:

                cls_res = cls.predict(crop_img,
                                      imgsz=pic_imgsz,
                                      conf=pic_conf,
                                      device=0,
                                      verbose=False)

                if cls_res[0].boxes is None:
                    continue

                lon, lat, alt, pitch, yaw, roll = infos[i]

                for k in range(len(cls_res[0].boxes)):
                    x1, y1, x2, y2 = map(int, cls_res[0].boxes.xyxy[k])

                    cx = rx + (x1 + x2)/2 * (side / pic_imgsz)
                    cy = ry + (y1 + y2)/2 * (side / pic_imgsz)

                    result = pixel_to_gps(
                        cx, cy, lon, lat, alt, pitch, yaw, roll)
                    if result is None:
                        continue
                    wlon, wlat = result

                    cls_id = int(cls_res[0].boxes.cls[k])

                    idx = update_wells(wlon, wlat, cls_id)

                    if SAVE_WELL_SNAPS and save_dir_exp:
                        cls_dir = save_dir_exp / f"cls_{cls_id}"
                        cls_dir.mkdir(parents=True, exist_ok=True)

                        cv2.imwrite(
                            str(cls_dir / f"{num_of_save}_{idx}.jpg"),
                            crop_img
                        )
                        num_of_save += 1

        torch.cuda.empty_cache()

def main():
    config = load_config()
    ground_test = config['ground_test']
    lat_target = config['throw_target']['lat']
    lon_target = config['throw_target']['lon']

    init_dirs()
    send(70,120)
    det = YOLO("weights/wellcut_1002.pt")
    cls = YOLO("weights/round1_1004_tqc.pt")

    det.fuse()
    try: det.model.half()
    except: pass

    cls.fuse()
    try: cls.model.half()
    except: pass

    t1 = threading.Thread(target=shm_thread)
    t2 = threading.Thread(target=infer_thread, args=(det, cls))

    t1.start()
    t2.start()

    t1.join()
    t2.join()

    choose_and_send()

if __name__ == "__main__":
    main()
