import cv2
import numpy as np
import math
from ultralytics import YOLO
import pytesseract
from collections import defaultdict
import datetime
import os
import torch
from pathlib import Path
import socket
from typing import List, Tuple

# ============================
# TX2-NX 4GB friendly settings
# ============================
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

lat = 30.5046127
lon = 120.1097193

# ---- Tunables ----
imgsz = 320
well_conf = 0.5
pic_conf = 0.75
num_conf = 80
scale_gps = 0.9
folder_img = "/home/tx2/Wintter/raw_pic"
folder_info = "/home/tx2/Wintter/info_to_ground"

# === Storage controls ===
SAVE_CROPS = True                                 # 是否保存 320×320 裁剪图
folder_cut = "/home/tx2/Wintter/send_to_ground"  # 裁剪图保存目录

SAVE_NUM_SNAPS = True                               # 是否保存“按识别数字分文件夹”的 warp/bin
runs_root = Path("runs/detect_num/exp")             # 会自动递增 exp、exp_2、exp_3…

# Memory guard rails
DET_BATCH = 6       # keep small to fit 4GB
CLS_BATCH = 12      # ditto
IMG_CHUNK = 12      # images per chunk

# OCR
OCR_WHITELIST = "0123456789"
OCR_PSM = 7
OCR_OEM = 1
OCR_GAUSS = 7
OCR_BLOCK = 15
OCR_C = 2

# Camera/frame geometry
img_cols = 1920
img_rows = 1080
well_width = 320
well_height = 320
cell_width = 1.263158
cell_height = 0.947368

SEND_PORT = 10000
send_server_addr = ('127.0.0.1', SEND_PORT)
response = ''

NUM_INFO = 10

# ------------ globals -------------
wells = []
num_of_cut = 0
num_of_save = 0
save_dir_exp: Path = None  # runs/detect_num/exp* 实际路径

# ------------ utils ---------------
def increment_path(path: Path, sep='_') -> Path:
    path = Path(path)
    if not path.exists():
        return path
    dirs = [d for d in path.parent.glob(f"{path.stem}*") if d.is_dir()]
    n = max([int(d.name.replace(path.stem + sep, '')) for d in dirs if sep in d.name] + [0]) + 1
    return path.parent / f"{path.stem}{sep}{n}"

def _init_save_dirs():
    global save_dir_exp
    if SAVE_CROPS:
        os.makedirs(folder_cut, exist_ok=True)
    if SAVE_NUM_SNAPS:
        save_dir_exp = increment_path(runs_root)
        save_dir_exp.mkdir(parents=True, exist_ok=True)

def enu_meters_per_deg(lat_deg: float):
    lat = math.radians(lat_deg)
    m_per_deg_lat = 111132.92 - 559.82 * math.cos(2 * lat) + 1.175 * math.cos(4 * lat)
    m_per_deg_lon = 111412.84 * math.cos(lat) - 93.5 * math.cos(3 * lat)
    return m_per_deg_lon, m_per_deg_lat

def dis(lon1, lat1, lon2, lat2):
    m_per_deg_lon, m_per_deg_lat = enu_meters_per_deg(lat1)
    return math.sqrt(((lon1 - lon2) * m_per_deg_lon) ** 2 + ((lat1 - lat2) * m_per_deg_lat) ** 2)

def read_info_from_png(info_path: str):
    img = cv2.imread(info_path)
    if img is None:
        return None
    h, w, ch = img.shape
    def read_seg(n):
        if w * ch <= n + 10:
            return ""
        row0 = img[0]
        bytes_list = [row0[i, 0] for i in range(n, n + 10)]
        return ''.join(chr(b) for b in bytes_list)
    segs = [read_seg(10 * k) for k in range(NUM_INFO)]
    if any(len(s) == 0 for s in segs):
        return None
    try:
        vals = [float(s.strip()) for s in segs]
        if len(vals) != NUM_INFO:
            return None
        return vals
    except:
        return None

def get_well_gps(lon, lat, alt, pitch, yaw, x_pic, y_pic):
    pitch_rad = math.radians(pitch)
    yaw_rad = math.radians(yaw)
    roll_rad = math.radians(5)

    xp = x_pic - img_cols / 2
    yp = img_rows / 2 - y_pic

    zc = alt / (
        yp * (cell_height / img_rows) * math.sin(pitch_rad)
        - xp * (cell_width / img_cols) * math.cos(pitch_rad) * math.sin(roll_rad)
        - math.cos(pitch_rad) * math.cos(roll_rad)
    )
    xc = - zc * xp * (cell_width / img_cols)
    yc = - zc * yp * (cell_height / img_rows)

    x = (
        xc * (math.cos(yaw_rad) * math.cos(roll_rad) + math.sin(yaw_rad) * math.sin(pitch_rad) * math.sin(roll_rad))
        + yc * math.sin(yaw_rad) * math.cos(pitch_rad)
        + zc * (math.cos(yaw_rad) * math.sin(roll_rad) - math.sin(yaw_rad) * math.sin(pitch_rad) * math.cos(roll_rad))
    )
    y = (
        xc * (-math.sin(yaw_rad) * math.cos(roll_rad) + math.cos(yaw_rad) * math.sin(pitch_rad) * math.sin(roll_rad))
        + yc * math.cos(yaw_rad) * math.cos(pitch_rad)
        + zc * (-math.sin(yaw_rad) * math.sin(roll_rad) - math.cos(yaw_rad) * math.sin(pitch_rad) * math.cos(roll_rad))
    )

    along = x * math.sin(yaw_rad) + y * math.cos(yaw_rad)
    cross = x * math.cos(yaw_rad) - y * math.sin(yaw_rad)
    along *= scale_gps
    x_corr = along * math.sin(yaw_rad) + cross * math.cos(yaw_rad)
    y_corr = along * math.cos(yaw_rad) - cross * math.sin(yaw_rad)

    m_per_deg_lon, m_per_deg_lat = enu_meters_per_deg(lat)
    _lon = 1.0 / m_per_deg_lon
    _lat = 1.0 / m_per_deg_lat

    well_lon = x_corr * _lon + lon
    well_lat = y_corr * _lat + lat
    return well_lon, well_lat

def init_addr(sockfd):
    sockfd.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  # 对socket的配置重用ip和端口号，只有服务端需要这个设置
    sockfd.bind(send_server_addr)
    sockfd.listen(40)

def send(lat, lon):
    global response
    while response == '':
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # 定义socket类型，网络通信，TCP
        init_addr(sock)
        connect_sock, client_addr = sock.accept()
        connect_sock.sendall(str(lat).encode())
        response = connect_sock.recv(100)
        print(response.decode())
        connect_sock.close()
    while response == b'get lat success   ':
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # 定义socket类型，网络通信，TCP
        init_addr(sock)
        connect_sock, client_addr = sock.accept()
        connect_sock.sendall(str(lon).encode())
        response = connect_sock.recv(100)
        print(response.decode())
        connect_sock.close()
    while response == b'get lon success   ':
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # 定义socket类型，网络通信，TCP
        init_addr(sock)
        connect_sock, client_addr = sock.accept()
        connect_sock.sendall('1'.encode())
        response = connect_sock.recv(100)
        print(response.decode())
        connect_sock.close()
    while response == b'send success      ':
        break

def is_angle_greater_than_180_counterclockwise(p1, p2, p3):
    vector12 = np.array(p2) - np.array(p1)
    vector23 = np.array(p3) - np.array(p2)
    cross_z = vector12[0] * vector23[1] - vector12[1] * vector23[0]
    return 1 if cross_z > 0 else 0

def get_number(img):
    config = f'--oem {OCR_OEM} --psm {OCR_PSM} -c tessedit_char_whitelist={OCR_WHITELIST}'
    data = pytesseract.image_to_data(img, config=config, lang='eng', output_type=pytesseract.Output.DICT)
    text_items = data.get('text', [])
    conf_items = data.get('conf', [])
    if not text_items:
        return "None", 0.0
    digits = ''.join(filter(str.isdigit, ''.join(text_items)))
    confidences = [int(c) for c, t in zip(conf_items, text_items) if str(t).isdigit() and str(c).isdigit()]
    avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
    if len(digits) == 2 and avg_conf >= num_conf:
        return digits, avg_conf
    return "None", avg_conf

def update_wells(new_lon, new_lat, dis_same_well=5, dis_dfrt_well=10):
    global wells
    if new_lon is None or new_lat is None:
        return None
    if not wells:
        wells.append({"lon": new_lon, "lat": new_lat, "points": [(new_lon, new_lat)], "per_num": defaultdict(list)})
        return 1
    min_dist = float('inf'); min_idx = -1
    for i, well in enumerate(wells):
        d = dis(new_lon, new_lat, well['lon'], well['lat'])
        if d < min_dist:
            min_dist = d; min_idx = i
    if min_dist <= dis_same_well:
        well = wells[min_idx]
        well['points'].append((new_lon, new_lat))
        lons, lats = zip(*well['points'])
        well['lon'] = sum(lons) / len(lons)
        well['lat'] = sum(lats) / len(lats)
        return min_idx + 1
    elif min_dist > dis_dfrt_well:
        wells.append({"lon": new_lon, "lat": new_lat, "points": [(new_lon, new_lat)], "per_num": defaultdict(list)})
        return len(wells)
    return None

def cut(img_bgr, results):
    """按第一阶段框裁剪 ；可选保存到 folder_cut。"""
    global num_of_cut
    out = []
    if results is None or results.boxes is None or len(results.boxes) == 0:
        return out
    img_h, img_w = img_bgr.shape[:2]
    for i in range(len(results.boxes)):
        x1, y1, x2, y2 = results.boxes.xyxy[i].cpu().numpy().astype(int)
        w = x2 - x1; h = y2 - y1
        if w < 50 or h < 50 or w > 150 or h > 150:
            continue
        x1 = max(x1 - 25, 0); y1 = max(y1 - 25, 0)
        w = min(w + 50, img_w - x1); h = min(h + 50, img_h - y1)
        if w <= 0 or h <= 0:
            continue
        crop = img_bgr[y1:y1+h, x1:x1+w]
        crop = cv2.resize(crop, (well_width, well_height), interpolation=cv2.INTER_LINEAR)
        if cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).mean() < 10:
            continue

        if SAVE_CROPS:
            out_name = os.path.join(folder_cut, f"{num_of_cut}.jpg")
            cv2.imwrite(out_name, crop)
            num_of_cut += 1

        out.append((crop, (x1, y1, w, h)))
    return out

def iter_image_paths(root_img, root_info):
    idx = 0
    while True:
        img_path = os.path.join(root_img, f"{idx}.jpg")
        info_path = os.path.join(root_info, f"{idx}.png")
        if not (os.path.exists(img_path) and os.path.exists(info_path)):
            break
        yield img_path, info_path
        idx += 2

def run_cls_batch(cls_model, batch_imgs, batch_meta, infos, parent_idx_offset=0):
    global num_of_save
    results = cls_model.predict(source=batch_imgs, imgsz=imgsz, conf=pic_conf,
                                device=0, half=True, verbose=False)
    for r, meta in zip(results, batch_meta):
        parent_idx, Rx, Ry, Rw, Rh = meta
        boxes = r.boxes
        kps = r.keypoints
        if boxes is None or len(boxes) == 0 or kps is None or len(kps.data) == 0:
            continue
        for k in range(len(boxes)):
            x1, y1, x2, y2 = boxes.xyxy[k].cpu().numpy().astype(int)
            h0, w0 = r.orig_img.shape[:2]
            x1 = max(0, min(x1, w0-1)); y1 = max(0, min(y1, h0-1))
            x2 = max(0, min(x2, w0));    y2 = max(0, min(y2, h0))
            crop_det = r.orig_img[y1:y2, x1:x2]
            kp = kps.data.cpu().numpy()[k]
            zjjg = [[xk, yk] for xk, yk, confk in kp]
            if not zjjg:
                continue

            # 反算子图中心到整图像素坐标
            xk = (zjjg[1][0] + zjjg[2][0] + zjjg[3][0] + zjjg[4][0]) / 4
            yk = (zjjg[1][1] + zjjg[2][1] + zjjg[3][1] + zjjg[4][1]) / 4
            cx_sub = int(xk * 0.634 + zjjg[0][0] * 0.366)
            cy_sub = int(yk * 0.634 + zjjg[0][1] * 0.366)
            cx = Rx + cx_sub * (Rw / 300.0)
            cy = Ry + cy_sub * (Rh / 300.0)

            detected_lon, detected_lat = get_well_gps(infos[parent_idx][8], infos[parent_idx][9], infos[parent_idx][2], infos[parent_idx][3], infos[parent_idx][4], cx, cy)
            idx = update_wells(detected_lon, detected_lat)

            if idx:
                # 关键点 → 透视
                zjjg_det = [[p[0] - x1, p[1] - y1] for p in zjjg]
                f = is_angle_greater_than_180_counterclockwise(zjjg_det[1], zjjg_det[2], zjjg_det[3])
                if f == 0:
                    pts1 = np.float32([zjjg_det[1], zjjg_det[4], zjjg_det[3], zjjg_det[2]])
                else:
                    pts1 = np.float32([zjjg_det[4], zjjg_det[1], zjjg_det[2], zjjg_det[3]])
                pts2 = np.float32([[0, 0], [100, 0], [100, 100], [0, 100]])
                M = cv2.getPerspectiveTransform(pts1, pts2)
                warped = cv2.warpPerspective(crop_det, M, (100, 100))

                # 轻量 OCR 预处理
                gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
                gray = cv2.GaussianBlur(gray, (OCR_GAUSS, OCR_GAUSS), 0)
                binimg = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY, OCR_BLOCK, OCR_C)
                cv2.waitKey(1)
                kernel = np.ones((11, 11), np.uint8)
                binimg = cv2.morphologyEx(binimg, cv2.MORPH_OPEN, kernel)
                kernel = np.ones((1, 1), np.uint8)
                binimg = cv2.erode(binimg, kernel, iterations=1) # 进行膨胀

                num_result, result_conf = get_number(binimg)

                if result_conf > num_conf:
                    wells[idx - 1]['per_num'][num_result].append((detected_lon, detected_lat))

                    # === 按“识别数字”分文件夹落盘 warp/bin ===
                    if SAVE_NUM_SNAPS and save_dir_exp is not None and num_result != "None":
                        try:
                            num_dir = save_dir_exp / str(num_result)  # 数字就是文件夹名
                            num_dir.mkdir(parents=True, exist_ok=True)
                            cv2.imwrite(str(num_dir / f"{num_of_save}_{idx}_warp.jpg"), warped)
                            cv2.imwrite(str(num_dir / f"{num_of_save}_{idx}_bin.jpg"),  binimg)
                            num_of_save += 1
                        except Exception as e:
                            print(f"[WARN] save snaps failed: {e}")

def process_chunk(det_model, cls_model, paths: List[Tuple[str, str]]):
    # Load only this chunk into RAM
    imgs = []
    infos = []
    img_paths = []
    for (img_p, info_p) in paths:
        img = cv2.imread(img_p)
        inf = read_info_from_png(info_p)
        if img is None or inf is None:
            continue
        imgs.append(img); infos.append(inf); img_paths.append(img_p)

    if not imgs:
        return

    # Stage-1 detection for this chunk
    det_results = det_model.predict(source=img_paths, imgsz=imgsz, conf=well_conf,
                                    device=0, half=True, verbose=False, batch=DET_BATCH)

    # Cut crops, push to cls in small batches
    crops = []
    metas = []
    for i, (img, res) in enumerate(zip(imgs, det_results)):
        c = cut(img, res)
        if not c:
            continue
        for (crop_img, rect_R) in c:
            crops.append(crop_img)
            metas.append((i, *rect_R))
            if len(crops) == CLS_BATCH:
                run_cls_batch(cls_model, crops, metas, infos, parent_idx_offset=0)
                crops, metas = [], []
    if crops:
        run_cls_batch(cls_model, crops, metas, infos, parent_idx_offset=0)

def choose_and_send():
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
        counts = {k: len(v) for k, v in w.get('per_num', {}).items()}
        main_num, main_cnt = (None, 0)
        if counts:
            filtered = {k: v for k, v in counts.items() if k not in (None, "None")} #排除None
            if filtered:
                main_num = max(filtered.items(), key=lambda x: x[1])[0]
            else:
                main_num = None
            main_cnt = counts[main_num]
        stats.append((idx, len(pts), avg_lon, avg_lat, main_num, main_cnt))
    if not stats:
        print("no main_num but sent")
        send(lat, lon)
        return True
    stats = sorted(stats, key=lambda x: x[1], reverse=True)[:3]

    # 输出三组信息
    print("Top 3 well groups:")
    for i, (idx, length, avg_lon, avg_lat, main_num, main_cnt) in enumerate(stats):
        print(f"Group {i+1}: avg_lon={avg_lon:.7f}, avg_lat={avg_lat:.7f}, main_num={main_num}, main_num_count={main_cnt}")

    nums = []
    for (idx, cnt, avg_lon, avg_lat, main_num, main_cnt) in stats:
        try:
            nums.append((int(main_num) if main_num not in (None,'None') else -1, idx, avg_lon, avg_lat, main_num))
        except:
            nums.append((-1, idx, avg_lon, avg_lat, main_num))
    nums_sorted = sorted(nums, key=lambda x: x[0])
    median_idx = 1 if len(nums_sorted) == 3 else 0
    _, widx, m_lon, m_lat, m_num = nums_sorted[median_idx]
    print(f"Send group: avg_lon={m_lon:.7f}, avg_lat={m_lat:.7f}, main_num={m_num}")
    send(m_lat, m_lon)
    return True

def main():
    start = datetime.datetime.now()
    _init_save_dirs()

    det = YOLO("weights/wellcut_1002.pt")
    det.fuse() #把模型的 Conv+BN 融合，推理更快
    try: det.model.half() #用 FP16 半精度运行，节省显存、提高速度，如果设备不支持就跳过
    except: pass

    cls = YOLO("weights/round2_0914.pt")
    cls.fuse()
    try: cls.model.half()
    except: pass

    # streaming over chunks to bound RAM
    chunk = []
    for pair in iter_image_paths(folder_img, folder_info):
        chunk.append(pair)
        #每次积累 IMG_CHUNK 对图像后，就调用 process_chunk() 处理这一批
        if len(chunk) == IMG_CHUNK:
            process_chunk(det, cls, chunk)
            #清空chunk,并释放内存
            chunk = []
            torch.cuda.empty_cache()
    #收尾处理
    if chunk:
        process_chunk(det, cls, chunk)
        torch.cuda.empty_cache()

    elapsed = (datetime.datetime.now() - start).total_seconds()
    print(f"[Total] streaming pipeline took {elapsed:.3f}s; wells={len(wells)}")
    ok = choose_and_send()
    print(f"send status: {ok}")

if __name__ == "__main__":
    main()
