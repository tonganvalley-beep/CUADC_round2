from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path

imgsz = 640
conf = 0.6

# 原图目录
base_dir = "/home/tttian/ultralytics-main/data/step1_0612/test/images/"

# 模型路径
model_path = "weights/step1_0717_960.pt"

# 图片最大序号
max_number = 1200

def increment_path(path: Path, exist_ok=False):
    path = Path(path)

    if exist_ok or not path.exists():
        return path

    i = 1
    while True:
        new_path = Path(f"{path}_{i}")
        if not new_path.exists():
            return new_path
        i += 1

# 保存目录
save_dir = increment_path(Path("runs/pose") / "predict")
save_dir.mkdir(parents=True, exist_ok=True)

def get_img(num_of_img):
    try:
        resource = f"{base_dir}{num_of_img}.jpg"
        img = cv2.imread(resource)
        return img
    except Exception as e:
        print(e)
        return None


def is_angle_greater_than_180_counterclockwise(p1, p2, p3):
    vector12 = np.array(p2) - np.array(p1)
    vector23 = np.array(p3) - np.array(p2)

    cross_z = (
        vector12[0] * vector23[1]
        - vector12[1] * vector23[0]
    )

    return 1 if cross_z > 0 else 0

def expand_quad_pixel(pts, expand=8):
    """
    四边形向外扩展固定像素

    pts:
        四个点，顺序:
        左上, 右上, 右下, 左下

    expand:
        向外扩展像素
    """

    pts = np.array(pts, dtype=np.float32)

    result = np.zeros_like(pts)

    # 中心
    center = np.mean(pts, axis=0)

    for i, p in enumerate(pts):

        # 当前点方向
        direction = p - center

        length = np.linalg.norm(direction)

        if length > 0:
            direction = direction / length

        # 沿远离中心方向移动固定距离
        result[i] = p + direction * expand

    return result

def run_detect(model, img, num_of_img):

    results = model.predict(
        source=img,
        imgsz=imgsz,
        conf=conf,
        verbose=False
    )

    if len(results) == 0:
        return

    result = results[0]

    well_folder = save_dir / "well"
    well_folder.mkdir(parents=True, exist_ok=True)

    cut_folder = save_dir / "cut"
    cut_folder.mkdir(parents=True, exist_ok=True)

    # 保存关键点检测结果图
    img_with_kpts = result.plot()

    cv2.imwrite(
        str(well_folder / f"{num_of_img}.jpg"),
        img_with_kpts
    )

    # 没有关键点
    if result.keypoints is None:
        return

    kpts = result.keypoints.data.cpu().numpy()

    print(f"Image {num_of_img}: detect {len(kpts)} wells")

    for obj_idx, kp in enumerate(kpts):

        # 至少需要5个点
        if len(kp) < 5:
            continue

        try:
            zjjg = []

            for p in kp:
                x, y, c = p
                zjjg.append([float(x), float(y)])

            # 0中心点
            center = zjjg[0]

            # 1~4角点
            p1 = zjjg[1]
            p2 = zjjg[2]
            p3 = zjjg[3]
            p4 = zjjg[4]

            # 保存关键点可视化
            debug_img = img.copy()

            cv2.circle(debug_img, (int(center[0]), int(center[1])), 8, (0, 0, 255), -1)

            for idx in range(1, 5):
                cv2.circle(
                    debug_img,
                    (int(zjjg[idx][0]), int(zjjg[idx][1])),
                    8,
                    (0, 255, 0),
                    -1
                )

                cv2.putText(
                    debug_img,
                    str(idx),
                    (int(zjjg[idx][0]), int(zjjg[idx][1])),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 0, 0),
                    2
                )

            cv2.imwrite(
                str(well_folder / f"{num_of_img}_{obj_idx}_points.jpg"),
                debug_img
            )

            # 判断顺逆时针
            f = is_angle_greater_than_180_counterclockwise(
                p1,
                p2,
                p3
            )

            if f == 0:
                pts1 = np.float32([
                    p1,
                    p4,
                    p3,
                    p2
                ])
            else:
                pts1 = np.float32([
                    p4,
                    p1,
                    p2,
                    p3
                ])
            
            pts1 = expand_quad_pixel(
                pts1,
                expand=3
            )
            
            pts2 = np.float32([
                [0, 0],
                [100, 0],
                [100, 100],
                [0, 100]
            ])

            M = cv2.getPerspectiveTransform(
                pts1,
                pts2
            )

            warped = cv2.warpPerspective(
                img,
                M,
                (100, 100)
            )

            # 保存到 well 文件夹
            cv2.imwrite(str(well_folder / f"{num_of_img}_{obj_idx}_warp.jpg"), warped)

			# 再保存到 cut 文件夹
            cv2.imwrite(str(cut_folder / f"{num_of_img}_{obj_idx}.jpg"), warped)

            print(
                f"saved {num_of_img}_{obj_idx}_warp.jpg"
            )

        except Exception as e:
            print(
                f"well {obj_idx} error:",
                e
            )


def main():

    model = YOLO(model_path)

    model.fuse()

    num_of_img = 0

    while True:

        img = get_img(num_of_img)

        if num_of_img > max_number:
            print("finished")
            break

        if img is None:
            print("null")
            num_of_img = num_of_img + 1
            continue
            
        run_detect(
            model,
            img,
            num_of_img
        )

        num_of_img += 1


if __name__ == "__main__":
    main()
