import math
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.spatial.transform import Rotation as R

img_cols = 1920 # 拍摄到的原图像素
img_rows = 1080
#_lon = 0.00001180199  # 0.0000113636 每向东走一米的经度增加
#_lat = 0.0000091256386  # 0.000009009 每向北走一米的纬度增加
cell_width = 0.899
cell_height = 0.473
scale_gps = 1.0

fx=2726.575028744192423
cx=928.327607790873003
fy=2727.167466485869227
cy=565.145908375505314
k1 = 0.08992784064861
k2 = 1.106881115498932
p1 = 0.000511898824711
p2 = 0.000987814578193
k3 = -9.232783177431058
DEG2RAD = np.pi / 180.0
RAD2DEG = 180.0 / np.pi
EARTH_RADIUS = 6378137.0  # WGS84
TARGET_LAT = 33.9487857
TARGET_LON = 117.1774021

def read_info(info_path):
    """读取 lon、lat、alt、pitch、yaw、roll；姿态角单位为弧度。"""
    try:
        values = [float(value) for value in info_path.read_text(encoding="utf-8-sig").split()]
    except (OSError, ValueError) as exc:
        raise ValueError(f"无法读取信息文件 {info_path}: {exc}") from exc

    if len(values) < 6:
        raise ValueError(
            f"信息文件 {info_path} 至少应包含 6 个数值，实际读取到 {len(values)} 个"
        )
    coordinate_values = values[:6]
    if not all(math.isfinite(value) for value in coordinate_values):
        raise ValueError(f"信息文件 {info_path} 包含非有限数值")

    return tuple(coordinate_values)


def read_image(image_path):
    """读取图片，并兼容 Windows 下包含中文的路径。"""
    try:
        image_data = np.fromfile(image_path, dtype=np.uint8)
    except OSError:
        return None
    return cv2.imdecode(image_data, cv2.IMREAD_COLOR)

def enu_meters_per_deg(lat_deg: float):
    """给定纬度，返回(每度经度对应的米, 每度纬度对应的米)。"""
    lat = math.radians(lat_deg)
    m_per_deg_lat = 111132.92 - 559.82*math.cos(2*lat) + 1.175*math.cos(4*lat)
    m_per_deg_lon = 111412.84*math.cos(lat) - 93.5*math.cos(3*lat)
    return m_per_deg_lon, m_per_deg_lat

# 通过天井在图片中像素坐标求解天井实际gps位置
def get_well_gps(lon, lat, alt, pitch, yaw, x_pic, y_pic):
    pitch_rad = math.radians(pitch+1.5)
    yaw_rad = math.radians(yaw)
    roll_rad = math.radians(-6.5)

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

def pixel_to_gps(u, v, lon0, lat0, rel_alt, pitch, yaw, roll=0.0):
    """将图像像素坐标换算为地面经纬度；姿态角单位为弧度。"""

    # 相机内参与畸变参数
    K = np.array([
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)
    D = np.array([k1, k2, p1, p2, k3], dtype=np.float64)

    distorted_point = np.array([[[u, v]]], dtype=np.float64)
    undistorted_point = cv2.undistortPoints(distorted_point, K, D)
    x_cam, y_cam = undistorted_point[0, 0]

    # OpenCV相机坐标：X向右、Y向下、Z沿镜头光轴。
    ray_camera = np.array([x_cam, y_cam, 1.0], dtype=np.float64)
    ray_camera /= np.linalg.norm(ray_camera)

    # 相机坐标 -> 机体FRD坐标：X向前、Y向右、Z向下。
    # 图像上方对应机头前方，因此forward=-camera_y。
    ray_body = np.array([
        -ray_camera[1],
         ray_camera[0],
         ray_camera[2]
    ], dtype=np.float64)
    ray_body /= np.linalg.norm(ray_body)

    # MAVROS local_position/pose为ENU约定，转换为NED/FRD约定。
    yaw_ned = math.pi / 2.0 - yaw
    pitch_ned = -pitch
    roll_ned = 0.0  # 云台补偿滚转，TXT中的机体roll不参与解算。

    body_to_ned = R.from_euler(
        'ZYX', [yaw_ned, pitch_ned, roll_ned]
    ).as_matrix()
    ray_ned = body_to_ned @ ray_body
    ray_north, ray_east, ray_down = ray_ned

    # 射线必须朝向地面。
    if ray_down <= 1e-8:
        return None

    scale = rel_alt / ray_down
    north = scale * ray_north
    east = scale * ray_east

    lat = lat0 + (north / EARTH_RADIUS) * RAD2DEG
    lon = lon0 + (
        east / (EARTH_RADIUS * math.cos(lat0 * DEG2RAD))
    ) * RAD2DEG

    return lon, lat

clicked_points = []
window_action = "next"


def onclick(event):
    global clicked_points, window_action
    button = getattr(event.button, "value", event.button)
    if button == 1 and event.xdata is not None and event.ydata is not None:
        point = (int(event.xdata), int(event.ydata))
        clicked_points.append(point)
        event.inaxes.plot(
            point[0], point[1], marker="+", color="red",
            markersize=14, markeredgewidth=2
        )
        event.canvas.draw_idle()
    elif button == 3:
        window_action = "next"
        plt.close()


def onkey(event):
    global clicked_points, window_action
    if event.key in (" ", "space", "right", "n", "escape"):
        window_action = "next"
        plt.close()
    elif event.key == "q":
        window_action = "quit"
        plt.close()

def get_pixel_points(img):
    global clicked_points, window_action
    clicked_points = []
    window_action = "next"
    fig, ax = plt.subplots(figsize=(16, 9)) # figsize 调整窗口大小 (单位是英寸)，dpi=100 时 16x9 ≈ 1600x900 像素
    manager = plt.get_current_fig_manager() # 获取窗口
    window = getattr(manager, "window", None)
    if hasattr(window, "wm_geometry"):
        window.wm_geometry("+0+0") # Tk 后端
    elif hasattr(window, "move"):
        window.move(0, 0) # Qt 后端
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax.set_title("左键：连续选择多个点    右键/空格/→/N/Esc：下一张    Q：结束")
    fig.canvas.mpl_connect('button_press_event', onclick)
    fig.canvas.mpl_connect('key_press_event', onkey)
    plt.show()
    return list(clicked_points), window_action

def main():
    data_dir = Path(__file__).resolve().parent / "pic_19"    
    img_dir = data_dir / "selected_pic"
    info_dir = data_dir / "info_to_ground"

    if not img_dir.is_dir() or not info_dir.is_dir():
        raise FileNotFoundError(
            f"找不到数据目录，请检查 {img_dir} 和 {info_dir}"
        )

    def numeric_sort_key(path):
        return (0, int(path.stem)) if path.stem.isdigit() else (1, path.stem)

    image_paths = sorted(img_dir.glob("*.jpg"), key=numeric_sort_key)
    if not image_paths:
        raise FileNotFoundError(f"{img_dir} 中没有找到 jpg 图片")

    coordinate_count = 0
    processed_image_count = 0
    for img_path in image_paths:
        info_path = info_dir / f"{img_path.stem}.txt"
        if not info_path.is_file():
            print(f"错误：{img_path.name} 没有对应的信息文件 {info_path.name}")
            continue

        img = read_image(img_path)
        if img is None:
            print(f"跳过 {img_path.name}：图片读取失败或文件已损坏")
            continue

        try:
            lon, lat, alt, pitch, yaw, roll = read_info(info_path)
        except ValueError as exc:
            print(f"错误：{exc}")
            continue
        processed_image_count += 1

        points, action = get_pixel_points(img)
        if points:
            for point_index, (x_pic, y_pic) in enumerate(points, start=1):
                result = pixel_to_gps(
                    x_pic, y_pic, lon, lat, alt, pitch, yaw, roll
                )
                if result is None:
                    continue
                well_lon, well_lat = result
                coordinate_count += 1
                m_per_deg_lon, m_per_deg_lat = enu_meters_per_deg(TARGET_LAT)
                east_error = (well_lon - TARGET_LON) * m_per_deg_lon
                north_error = (well_lat - TARGET_LAT) * m_per_deg_lat
                distance_error = math.hypot(east_error, north_error)
                print(
                    f"{img_path.name} 点{point_index}: "
                    f"经度={well_lon}, 纬度={well_lat}, "
                    f"距目标={distance_error:.3f}米 "
                    f"(东西={east_error:+.3f}米, 南北={north_error:+.3f}米)"
                )

        if action == "quit":
            break

    print(
        f"完成：共处理 selected_pic 中 {processed_image_count} 张图片，"
        f"得到 {coordinate_count} 个经纬度坐标。"
    )

if __name__ == "__main__":
    main()
