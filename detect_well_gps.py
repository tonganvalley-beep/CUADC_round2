import cv2
import numpy as np
import math
from ultralytics import YOLO
from collections import defaultdict
import datetime
import time
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
well_imgsz = 320
pic_imgsz = 640
well_conf = 0.5
pic_conf = 0.75
scale_gps = 0.9
folder_img = "/media/tttian/tttian/对地/1004/pic2/raw_pic"
folder_info = "/media/tttian/tttian/对地/1004/pic2/info_to_ground"

SAVE_CROPS = True
folder_cut = "/media/tttian/tttian/对地/1004/pic2/send_to_ground"

SAVE_WELL_SNAPS = True
runs_root = Path("runs/detect_well/exp")

# Memory guard rails
DET_BATCH = 6                         #检测批次大小
CLS_BATCH = 12                        #分类批次大小
IMG_CHUNK = 12                        #每次处理的图片大小

# Camera/frame geometry
img_cols = 1920
img_rows = 1080
well_width = 640
well_height = 640
cell_width = 1.263158
cell_height = 0.947368

SEND_PORT = 10000
send_server_addr = ('127.0.0.1', SEND_PORT)
response = ''

NUM_INFO = 10

# ------------ globals -------------
wells = []
num_of_save = 0
num_of_cut = 0
save_dir_exp: Path = None

# ------------ utils ---------------
def increment_path(path: Path, sep='_') -> Path:  #返回值类型注释 表示该函数的返回值是一个Path类型的对象
    """自动递增路径：若 exp 已存在则生成 exp_1，若 exp_1 也存在则生成 exp_2，依此类推"""
    path = Path(path)
    if not path.exists():
        return path
    dirs = [d for d in path.parent.glob(f"{path.stem}*") if d.is_dir()]  # 找到同级目录下所有同名前缀的文件夹
    n = max([int(d.name.replace(path.stem + sep, '')) for d in dirs if sep in d.name] + [0]) + 1
    return path.parent / f"{path.stem}{sep}{n}"

def _init_save_dirs():
    """根据配置开关创建裁剪图和检测结果的保存文件夹"""
    global save_dir_exp
    if SAVE_CROPS:
        os.makedirs(folder_cut, exist_ok=True)
    if SAVE_WELL_SNAPS:
        save_dir_exp = increment_path(runs_root)
        save_dir_exp.mkdir(parents=True, exist_ok=True)

def enu_meters_per_deg(lat_deg: float):
    """根据给定纬度计算该处每经度、每纬度分别对应多少米（考虑地球椭球形状）"""
    lat = math.radians(lat_deg)
    m_per_deg_lat = 111132.92 - 559.82 * math.cos(2 * lat) + 1.175 * math.cos(4 * lat)
    m_per_deg_lon = 111412.84 * math.cos(lat) - 93.5 * math.cos(3 * lat)
    return m_per_deg_lon, m_per_deg_lat

def dis(lon1, lat1, lon2, lat2):
    """计算两个GPS坐标之间的平面距离（单位：米）"""
    m_per_deg_lon, m_per_deg_lat = enu_meters_per_deg(lat1)
    return math.sqrt(((lon1 - lon2) * m_per_deg_lon) ** 2 + ((lat1 - lat2) * m_per_deg_lat) ** 2)

def read_info_from_png(info_path: str):
    """从C++端生成的PNG图片第一行像素中解码出NUM_INFO个浮点数（包含高度、姿态、速度等）"""
    img = cv2.imread(info_path)  #之前没主义，但是我估计前面pic_get_gps.cpp里面已经把图片信息编码进了像素值里面
    if img is None:
        return None
    h, w, ch = img.shape  #高度，宽度，通道数
    def read_seg(n):          #python里怎么还有嵌套定义的，read_seg这个函数定义在一个函数的内部
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

def get_well_gps(lon, lat, alt, pitch, yaw, x_pic, y_pic): #略
    """将图像上的像素坐标 (x_pic, y_pic) 反算为真实GPS经纬度（利用飞机姿态和高度做空间几何变换）"""
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
    """配置服务端socket：允许端口复用，绑定地址，开始监听"""
    sockfd.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  # 对socket的配置重用ip和端口号，只有服务端需要这个设置
    sockfd.bind(send_server_addr)
    sockfd.listen(40)

def send(lat, lon):
    """通过TCP分三步将纬度、经度、结束标志依次发送给C++端，并等待确认回复"""
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

def update_wells(new_lon, new_lat, dis_same_well=10, dis_dfrt_well=10):
    """管理井列表：若新坐标靠近已知井(≤dis_same_well米)则归入该井并更新平均位置，若远离所有已知井(>dis_dfrt_well米)则创建新井"""
    global wells
    if new_lon is None or new_lat is None:
        return None
    if not wells:  #如果该列表为空
        wells.append({"lon": new_lon, "lat": new_lat, "points": [(new_lon, new_lat)], "per_class": defaultdict(list)})
        return 1
    min_dist = float('inf'); min_idx = -1  #float('inf')表示无穷大，任何数都可以更新它，便于求解最小
    for i, well in enumerate(wells): #enumerate 自动返回元素和元素对应的下标，当然其实是下标在前，元素在后
        d = dis(new_lon, new_lat, well['lon'], well['lat'])
        if d < min_dist:
            min_dist = d; min_idx = i
    if min_dist <= dis_same_well:
        well = wells[min_idx] #想起来python列表索引这回事 -1表示从最后往前数的第一个 也即将元素加到末尾
        well['points'].append((new_lon, new_lat)) 
        lons, lats = zip(*well['points'])         #zip就是把well的points坐标拆开来，第一个元素全部归为一组即为经度，第二个元素全部归为一组，即为纬度
        well['lon'] = sum(lons) / len(lons)  #平均经度
        well['lat'] = sum(lats) / len(lats)  #平均纬度
        return min_idx + 1
    elif min_dist > dis_dfrt_well:
        wells.append({"lon": new_lon, "lat": new_lat, "points": [(new_lon, new_lat)], "per_class": defaultdict(list)})
        return len(wells)
    return None

def cut(img_bgr, results, extra_pad=15):
    """根据YOLO检测框在原图上裁剪目标区域，补齐为正方形，可选保存裁剪图"""
    global num_of_cut
    out = []
    if results is None or results.boxes is None or len(results.boxes) == 0:
        return out

    img_h, img_w = img_bgr.shape[:2] #shape 宽，高，BGR通道数 [:2]意为只取前两个
    for i in range(len(results.boxes)):  #boxes：YOLO的相框 xyxy获取x1,y1,x2,y2
        x1, y1, x2, y2 = results.boxes.xyxy[i].cpu().numpy().astype(int)
        w = x2 - x1
        h = y2 - y1
        if w <= 0 or h <= 0:
            continue

        side = max(w, h) + 2 * extra_pad   # 上下左右各扩展 extra_pad
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        new_x1 = cx - side // 2
        new_y1 = cy - side // 2
        new_x2 = new_x1 + side
        new_y2 = new_y1 + side

        crop_x1 = max(new_x1, 0)
        crop_y1 = max(new_y1, 0)
        crop_x2 = min(new_x2, img_w)
        crop_y2 = min(new_y2, img_h)

        crop = img_bgr[crop_y1:crop_y2, crop_x1:crop_x2]

        # --- 创建全黑正方形背景 ---
        square = np.zeros((side, side, 3), dtype=np.uint8)

        offset_x = crop_x1 - new_x1  # 如果越界，偏移 > 0
        offset_y = crop_y1 - new_y1
        square[offset_y:offset_y+crop.shape[0], offset_x:offset_x+crop.shape[1]] = crop

        square = cv2.resize(square, (well_width, well_height), interpolation=cv2.INTER_LINEAR)

        if SAVE_CROPS:
            out_name = os.path.join(folder_cut, f"{num_of_cut}.jpg")
            cv2.imwrite(out_name, square)
            num_of_cut += 1

        out.append((square, (new_x1, new_y1, side, side)))
    return out

def iter_image_paths(root_img, root_info):
    """生成器：按序号0,1,2,... 依次产出同时存在的 .jpg 和 .png 路径对，找不到就停止"""
    idx = 0
    while True:
        img_path = os.path.join(root_img, f"{idx}.jpg")
        info_path = os.path.join(root_info, f"{idx}.png")
        if not (os.path.exists(img_path) and os.path.exists(info_path)): 
            break
        yield img_path, info_path   #yield 普通函数变成生成器，每需要的时候就吐出一个jpg和png的路径
        idx += 1

def run_cls_batch(cls_model, batch_imgs, batch_meta, infos):
    """对一批裁剪图运行分类模型，将分类结果反算GPS坐标并合并到井列表"""
    global num_of_save
    results = cls_model.predict(source=batch_imgs, imgsz=pic_imgsz, conf=pic_conf,
                                device=0, half=True, verbose=False) #yolo调用 调用的结果返回到列表里面
    for r, meta in zip(results, batch_meta):
        parent_idx, Rx, Ry, Rw, Rh = meta
        boxes = r.boxes
        if boxes is None or len(boxes) == 0:
            continue
        for k in range(len(boxes)):
            x1, y1, x2, y2 = boxes.xyxy[k].cpu().numpy().astype(int)
            h0, w0 = r.orig_img.shape[:2]
            x1 = max(0, min(x1, w0-1)); y1 = max(0, min(y1, h0-1))
            x2 = max(0, min(x2, w0));    y2 = max(0, min(y2, h0))
            crop_det = r.orig_img[y1:y2, x1:x2]
            cx = Rx + (x1 + x2) * 0.5 * (Rw / well_width)
            cy = Ry + (y1 + y2) * 0.5 * (Rh / well_height)
            detected_lon, detected_lat = get_well_gps(infos[parent_idx][8], infos[parent_idx][9], infos[parent_idx][2], infos[parent_idx][3], infos[parent_idx][4], cx, cy)
            idx = update_wells(detected_lon, detected_lat)
            if idx:
                wells[idx - 1]['per_class'][int(boxes.cls[k].item())].append((detected_lon, detected_lat))#boxes.cls[k].item() 根据检测的分类结果给出名字
                if SAVE_WELL_SNAPS and save_dir_exp is not None:
                    try:
                        cls_dir = save_dir_exp / f"cls_{int(boxes.cls[k].item())}"
                        cls_dir.mkdir(parents=True, exist_ok=True)
                        cv2.imwrite(str(cls_dir / f"p{num_of_save}_{idx}.jpg"), crop_det)
                        num_of_save += 1
                    except Exception as e:
                        print(f"[WARN] save snaps failed: {e}")

def process_chunk(det_model, cls_model, paths: List[Tuple[str, str]]):
    """处理一批图片：先用det模型找井→裁剪→再用cls模型分类→合并井坐标"""
    imgs = []
    infos = []
    img_paths = []
    for (img_p, info_p) in paths:   #输入一批图片 到imgs列表里
        img = cv2.imread(img_p)
        inf = read_info_from_png(info_p)
        if img is None or inf is None:
            continue
        imgs.append(img); infos.append(inf); img_paths.append(img_p)

    if not imgs:
        return

    det_results = det_model.predict(source=img_paths, imgsz=well_imgsz, conf=well_conf,
                                    device=0, half=True, verbose=False, batch=DET_BATCH)#检测天井 保存结果

    crops = []
    metas = []
    for i, (img, res) in enumerate(zip(imgs, det_results)):
        c = cut(img, res)  #调用前面的工具，裁出来天井
        if not c:  #没有裁出来天井就继续下一张图片
            continue
        for (crop_img, rect_R) in c:
            crops.append(crop_img)
            metas.append((i, *rect_R))
            if len(crops) == CLS_BATCH:
                run_cls_batch(cls_model, crops, metas, infos)
                crops, metas = [], []
    if crops:
        run_cls_batch(cls_model, crops, metas, infos)

def choose_and_send(cls):
    """从所有检测到的井中选出被检测次数最多的那口，将其平均GPS坐标发送给C++端"""
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

def main():
    """主流程：加载模型 → 逐批读取图片 → 检测井并分类 → 选出最佳结果发送"""
    start = datetime.datetime.now()
    _init_save_dirs()

    det = YOLO("weights/wellcut_1002.pt")
    det.fuse()
    try: det.model.half()
    except: pass

    cls = YOLO("weights/round1_1004_tqc.pt")
    cls.fuse()
    try: cls.model.half()
    except: pass

    chunk = []
    for pair in iter_image_paths(folder_img, folder_info):
        chunk.append(pair)
        if len(chunk) == IMG_CHUNK:
            process_chunk(det, cls, chunk)
            chunk = []
            torch.cuda.empty_cache()
    if chunk:
        process_chunk(det, cls, chunk)
        torch.cuda.empty_cache()

    elapsed = (datetime.datetime.now() - start).total_seconds()
    print(f"[Total] streaming pipeline took {elapsed:.3f}s; wells={len(wells)}")
    ok = choose_and_send(cls)
    print(f"send status: {ok}")

if __name__ == "__main__":
    main()
