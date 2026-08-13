"""
相机标定脚本
python calib.py
"""

import cv2
import numpy as np
import os
import yaml
import shutil
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
raw_path = BASE_DIR / "calib_images"
config_path = BASE_DIR / "config" / "chess_param.yml"
delete_threshold = 0.5
show_preview = os.environ.get("CALIB_SHOW_PREVIEW", "1").strip().lower() not in {
    "0", "false", "no", "off"
}

class CameraCalibration:
    def __init__(self, board_size, diameter, distance,  image_dir="calib_images"):
        """
        初始化标定参数
        
        Args:
            board_size: 标定板尺寸 (列数, 行数)
            diameter: 圆点直径，单位mm
            distance: 圆心间距，单位mm
            image_dir: 图片目录
        """
        self.board_size = board_size
        self.diameter = diameter
        self.square_size = distance  # 间距 = 直径 × 2
        self.image_dir = Path(image_dir)
        
        # 准备世界坐标系中的圆点三维坐标 (Z=0平面)
        self.objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2)
        self.objp *= self.square_size
        
        # 存储所有图像的角点坐标
        self.objpoints = []  # 三维点
        self.imgpoints = []  # 二维点
        self.image_names = []  # 图片名称
        
        # 标定结果
        self.camera_matrix = None
        self.dist_coeffs = None
        self.rvecs = None
        self.tvecs = None
        self.rms_error = None
        self.image_size = None

        # 针对低照度图像中的黑色圆点配置检测器。OpenCV 5 的默认参数
        # 在本项目图片（整体灰度较低）上可能无法产生任何候选圆斑。
        blob_params = cv2.SimpleBlobDetector_Params()
        blob_params.minThreshold = 5
        blob_params.maxThreshold = 240
        blob_params.thresholdStep = 5
        blob_params.filterByArea = True
        blob_params.minArea = 300
        blob_params.maxArea = 10000
        blob_params.filterByCircularity = True
        blob_params.minCircularity = 0.45
        blob_params.filterByConvexity = False
        blob_params.filterByInertia = False
        blob_params.filterByColor = True
        blob_params.blobColor = 0
        self.blob_detector = cv2.SimpleBlobDetector_create(blob_params)
    
    def find_circles_in_images(self):
        """
        在所有图片中检测圆点标定板
        """
        # 获取所有图片，兼容 Windows 路径、中文路径和大小写扩展名
        supported_extensions = {".jpg", ".jpeg", ".png"}
        image_files = sorted(
            path for path in self.image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in supported_extensions
        ) if self.image_dir.is_dir() else []
        
        if not image_files:
            print(f"[FAIL] 未在 {self.image_dir} 中找到图片！")
            return False
        
        successful_count = 0
        
        for idx, fname in enumerate(image_files, 1):
            # np.fromfile + imdecode 可避免部分 Windows OpenCV 版本无法读取中文路径
            try:
                img_data = np.fromfile(fname, dtype=np.uint8)
                img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
            except (OSError, ValueError):
                img = None
            if img is None:
                print(f"[FAIL] [{idx}/{len(image_files)}] 无法读取: {fname}")
                continue
            
            # 保存图像尺寸
            if self.image_size is None:
                self.image_size = (img.shape[1], img.shape[0])
            elif self.image_size != (img.shape[1], img.shape[0]):
                print(
                    f"[SKIP] [{idx}/{len(image_files)}] {fname.name} - "
                    f"分辨率不一致，期望{self.image_size}，实际{(img.shape[1], img.shape[0])}"
                )
                continue
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # 查找圆点
            ret, centers = cv2.findCirclesGrid(
                gray, 
                self.board_size,
                flags=cv2.CALIB_CB_SYMMETRIC_GRID |
                cv2.CALIB_CB_CLUSTERING,
                blobDetector=self.blob_detector,
            )
            
            if ret:
                self.objpoints.append(self.objp.copy())
                self.imgpoints.append(centers)
                self.image_names.append(fname.name)
                successful_count += 1
                print(f"[OK] [{idx}/{len(image_files)}] {fname.name}")
                
                # 绘制检测到的圆点
                if show_preview:
                    cv2.drawChessboardCorners(img, self.board_size, centers, ret)
                    cv2.imshow("findCorners", img)
                    cv2.waitKey(1)
            else:
                print(f"[SKIP] [{idx}/{len(image_files)}] {fname.name} - 未检测到标定板")
        
        print(f"\n成功检测: {successful_count}/{len(image_files)} 张")
        
        # 关闭所有窗口
        if show_preview:
            cv2.destroyAllWindows()
        
        if successful_count < 3:
            print(f"[FAIL] 有效图片数量不足（至少需要3张，当前{successful_count}张）")
            return False
        
        return True
    
    def calibrate(self):
        """
        执行相机标定
        """
        if not self.objpoints or not self.imgpoints:
            print("[FAIL] 没有可用的角点数据！")
            return False

        # 执行标定
        self.rms_error, self.camera_matrix, self.dist_coeffs, self.rvecs, self.tvecs = \
            cv2.calibrateCamera(
                self.objpoints,
                self.imgpoints,
                self.image_size,
                None,
                None
            )
        
        return True
    
    def print_per_image_error(self):
        """
        输出每张图片的重投影误差
        """

        print("\n" + "=" * 70)
        print("每张图片重投影误差")
        print("=" * 70)

        rms_list = []

        for i in range(len(self.objpoints)):
        
            imgpoints2, _ = cv2.projectPoints(
                self.objpoints[i],
                self.rvecs[i],
                self.tvecs[i],
                self.camera_matrix,
                self.dist_coeffs
            )

            # OpenCV 4/5 返回的点数组形状可能分别为 (N, 1, 2) 或
            # (N, 2)。统一展开后计算每个点的 RMS，避免 cv2.norm 类型不匹配。
            observed = np.asarray(self.imgpoints[i], dtype=np.float64).reshape(-1, 2)
            projected = np.asarray(imgpoints2, dtype=np.float64).reshape(-1, 2)
            delta = observed - projected
            err = np.sqrt(np.mean(np.sum(delta * delta, axis=1)))

            rms_list.append(
                (
                    self.image_names[i],
                    err
                )
            )

        rms_list.sort(
            key=lambda x: x[1],
            reverse=True
        )

        for name, err in rms_list:

            mark = ""

            if err > 1.0:
                mark = "  <-- 建议删除"

            elif err > 0.8:
                mark = "  <-- 偏大"

            print(
                f"{name:<30}"
                f"{err:.4f} px"
                f"{mark}"
            )

        print("=" * 70)
        
    def print_results(self):
        """
        打印标定结果
        """
        # 禁用科学计数法
        np.set_printoptions(suppress=True, precision=15, floatmode='maxprec_equal')

        print(f"\n{'='*60}")
        print("标定结果")
        print(f"{'='*60}\n")

        print(f"相机内参:")
        for i in range(3):
            # 使用format函数禁用科学计数法
            row_vals = []
            for val in self.camera_matrix[i]:
                if val == 0:
                    row_vals.append("0")
                else:
                    # 使用format禁用科学计数法，然后去除尾部零
                    s = format(val, '.15f').rstrip('0').rstrip('.')
                    row_vals.append(s)
            row_str = ", ".join(row_vals)
            print(f"[{row_str}]")

        print(f"\n畸变系数:")
        dist_vals = []
        for val in self.dist_coeffs[0]:
            # 使用format函数禁用科学计数法，然后去除尾部零
            s = format(val, '.15f').rstrip('0').rstrip('.')
            dist_vals.append(s)
        dist_str = ", ".join(dist_vals)
        print(f"[{dist_str}]")

        # 打印重投影误差和质量评估
        print(f"\nRMS 重投影误差: {self.rms_error:.4f} 像素")

        if self.rms_error < 0.3:
            print("  -> 标定质量: 优秀 [OK]")
        elif self.rms_error < 0.5:
            print("  -> 标定质量: 良好，可用 [OK]")
        elif self.rms_error < 0.8:
            print("  -> 标定质量: 一般，建议检查高误差图片")
        else:
            print("  -> 标定质量: 较差，建议重新采集 [WARN]")
            
    def remove_bad_images(self, threshold=0.8):
        """
        将重投影误差大于 threshold 的图片移动到 bad_images 目录
        """

        save_dir = self.image_dir / "bad_images"
        os.makedirs(save_dir, exist_ok=True)

        removed = 0

        print("\n开始筛选误差较大的图片...")

        for i in range(len(self.objpoints)):

            imgpoints2, _ = cv2.projectPoints(
                self.objpoints[i],
                self.rvecs[i],
                self.tvecs[i],
                self.camera_matrix,
                self.dist_coeffs
            )

            observed = np.asarray(self.imgpoints[i], dtype=np.float64).reshape(-1, 2)
            projected = np.asarray(imgpoints2, dtype=np.float64).reshape(-1, 2)
            delta = observed - projected
            err = np.sqrt(np.mean(np.sum(delta * delta, axis=1)))

            if err > threshold:

                src = self.image_dir / self.image_names[i]

                dst = save_dir / self.image_names[i]

                if os.path.exists(src):

                    shutil.move(src, dst)

                    print(
                        f"删除(移动): {self.image_names[i]}   "
                        f"RMS={err:.4f}"
                    )

                    removed += 1

        print(f"\n共移动 {removed} 张图片")


def load_config(config_file=config_path):
    """
    从yaml文件加载标定板参数
    """
    if not os.path.exists(config_file):
        print(f"[FAIL] 错误: 配置文件不存在: {config_file}")
        return None
    
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        if config is None:
            print(f"[FAIL] 错误: 配置文件为空: {config_file}")
            return None
        
        # 读取参数（不提供默认值）
        row = config.get('row')
        col = config.get('col') or config.get('column')  # 兼容 col 和 column
        diameter = config.get('diameter')
        distance = config.get('distance')
        
        # 检查必需参数
        if row is None:
            print(f"[FAIL] 错误: 配置文件缺少 'row' 参数")
            return None
        if col is None:
            print(f"[FAIL] 错误: 配置文件缺少 'col' 或 'column' 参数")
            return None
        if diameter is None:
            print(f"[FAIL] 错误: 配置文件缺少 'diameter' 参数")
            return None
        if distance is None:
            print(f"[FAIL] 错误: 配置文件缺少 'distance' 参数")
            return None
        
        return (col, row), diameter, distance
        
    except Exception as e:
        print(f"[FAIL] 错误: 读取配置文件失败: {e}")
        return None


def main():
    """
    主函数
    """
    print("\n" + "="*50)
    print("圆点标定板相机标定")
    print("="*50)
    
    # 从配置文件加载参数
    config_result = load_config(config_path)
    if config_result is None:
        print(f"\n请检查配置文件: {config_path}")
        print("需要包含以下参数:")
        print("  - row: 标定板行数")
        print("  - col: 标定板列数")
        print("  - diameter: 圆点直径(mm)")
        print("  - distance: 圆心间距(mm)")
        return
    
    board_size, diameter, distance = config_result
    
    print(f"\n标定板参数:")
    print(f"  行数 x 列数: {board_size[1]} x {board_size[0]}")
    print(f"  圆点直径: {diameter} mm")
    print(f"  圆点间距: {distance} mm")
    print()
    
    # 创建标定对象
    calib = CameraCalibration(
        board_size=board_size,
        diameter=diameter,
        distance=distance,
        image_dir=raw_path
    )
    
    # 检测圆点
    if not calib.find_circles_in_images():
        return
    
    # 执行标定
    if not calib.calibrate():
        return
    
    # 打印结果
    calib.print_results()
    calib.print_per_image_error()
    calib.remove_bad_images(delete_threshold)


if __name__ == "__main__":
    main()
