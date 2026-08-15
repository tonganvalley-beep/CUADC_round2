import socket
import cv2
import numpy as np
import struct
from pathlib import Path
import time
import threading


SKY_IP = "aab876d14723b375.natapp.cc"  # IP地址
RCV_PORT = 61200                       # 端口
RECONNECT_WAIT_S = 2                   # 断线重连等待

SAVE_ROOT = Path("D:/Duidi_pics") / "pics"


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


SAVE_DIR = increment_path(SAVE_ROOT)
SAVE_DIR.mkdir(parents=True, exist_ok=True)

print(f"[INFO] images will be saved to: {SAVE_DIR}")


# 全局变量

# 最新接收到的图片
latest_img = None

# 保护 latest_img
img_lock = threading.Lock()

# 控制程序运行
running = True

# 图片编号
num = 0


# 精确接收 n 字节

def recv_exact(sock, n):
    data = b''

    while len(data) < n:

        chunk = sock.recv(n - len(data))

        if not chunk:
            return None

        data += chunk

    return data


# 接收一张图片

def get_img(sock):

    # 先收 4 字节无符号整型（网络序）的图片长度

    size_bytes = recv_exact(sock, 4)

    if not size_bytes:
        return None

    img_size = struct.unpack('!I', size_bytes)[0]

    # 防止异常数据
    if img_size == 0 or img_size > 100_000_000:

        print(f"[WARN] invalid image size: {img_size}")

        return None

    # 收图像二进制

    img_data = recv_exact(sock, img_size)

    if not img_data:
        return None

    # OpenCV解码图像

    arr = np.frombuffer(
        img_data,
        dtype=np.uint8
    )

    img = cv2.imdecode(
        arr,
        cv2.IMREAD_COLOR
    )

    if img is None:

        print("[WARN] cv2.imdecode failed")

        return None

    return img


# 网络接收线程

def receive_thread():

    global latest_img
    global running
    global num

    while running:

        sock = socket.socket(
            socket.AF_INET,
            socket.SOCK_STREAM
        )

        try:

            print(
                f"[INFO] connecting to "
                f"{SKY_IP}:{RCV_PORT} ..."
            )

            sock.connect(
                (SKY_IP, RCV_PORT)
            )

            print("[INFO] connected")

            # 收流循环

            while running:

                img = get_img(sock)

                # 接收失败 / 对方断开

                if img is None:

                    print(
                        "[WARN] image recv failed/closed; "
                        "reconnect..."
                    )

                    break

                # 保存图片

                save_path = SAVE_DIR / f"{num}.jpg"

                success = cv2.imwrite(
                    str(save_path),
                    img
                )

                if success:

                    print(
                        f"[OK] saved {save_path}"
                    )

                else:

                    print(
                        f"[WARN] failed to save "
                        f"{save_path}"
                    )

                # 更新最新图片
                #
                # 注意：
                # 这里只更新最新的一张。
                # 不在这里调用 imshow。

                with img_lock:

                    latest_img = img

                num += 1

        except OSError as e:

            print(
                f"[WARN] connection error: {e}"
            )

        finally:

            try:
                sock.close()

            except Exception:
                pass

        # 断线重连

        if running:

            print(
                f"[INFO] reconnecting in "
                f"{RECONNECT_WAIT_S}s..."
            )

            time.sleep(
                RECONNECT_WAIT_S
            )


# 主程序

def main():

    global running

    # 创建 OpenCV 窗口

    cv2.namedWindow(
        "recv",
        cv2.WINDOW_NORMAL
    )

    # 设置窗口置顶
    cv2.setWindowProperty(
        "recv",
        cv2.WND_PROP_TOPMOST,
        1
    )

    # 启动网络接收线程

    t = threading.Thread(
        target=receive_thread,
        daemon=True
    )

    t.start()

    print("[INFO] receiver thread started")

    # 主线程只负责 OpenCV 窗口

    while True:

        # 获取最新图片

        with img_lock:

            if latest_img is not None:

                img = latest_img.copy()

            else:

                img = None

        # 显示图片

        if img is not None:

            cv2.imshow(
                "recv",
                img
            )

        key = cv2.waitKey(300) & 0xFF

        # ESC退出
        if key == 27:

            print("[INFO] ESC pressed")

            running = False

            break

        # q退出
        if key == ord('q'):

            print("[INFO] q pressed")

            running = False

            break

    cv2.destroyAllWindows()

    print("[INFO] program exited")


# 程序入口

if __name__ == "__main__":

    main()