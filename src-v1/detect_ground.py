import socket
import cv2
import numpy as np
import struct
from pathlib import Path
import time

SKY_IP = "47.97.153.27"  # IP地址
RCV_PORT = 61200         # 端口
RECONNECT_WAIT_S = 2     # 断线重连等待
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

def recv_exact(sock, n):
    data = b''
    while len(data) < n:
        chunk = sock.recv(n - len(data))
        if not chunk:
            return None
        data += chunk
    return data

def get_img(sock):
    # 先收 4 字节无符号整型(网络序)的长度
    size_bytes = recv_exact(sock, 4)
    if not size_bytes:
        return None
    img_size = struct.unpack('!I', size_bytes)[0]
    if img_size == 0 or img_size > 100_000_000:
        return None

    # 收图像二进制
    img_data = recv_exact(sock, img_size)
    if not img_data:
        return None

    # 解码 opencv 图像
    arr = np.frombuffer(img_data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img

def main():
    num = 0
    cv2.namedWindow("recv", cv2.WINDOW_NORMAL)
    # 设置窗口置顶
    cv2.setWindowProperty("recv", cv2.WND_PROP_TOPMOST, 1)
    while True:  # 重连大循环
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            print(f"[INFO] connecting to {SKY_IP}:{RCV_PORT} ...")
            sock.connect((SKY_IP, RCV_PORT))
            print("[INFO] connected")
        except OSError as e:
            print(f"[WARN] connect failed: {e}; retry in {RECONNECT_WAIT_S}s")
            time.sleep(RECONNECT_WAIT_S)
            continue

        # 收流循环
        try:
            while True:
                img = get_img(sock)
                if img is None:
                    print("[WARN] image recv failed/closed; reconnect...")
                    break
                save_path = SAVE_DIR / f"{num}.jpg"
                cv2.imwrite(str(save_path), img)
                print(f"[OK] saved {save_path}")
                cv2.imshow("recv", img)
                cv2.waitKey(300)
                num += 1
        finally:
            try:
                sock.close()
            except Exception:
                pass
            time.sleep(RECONNECT_WAIT_S)

if __name__ == "__main__":
    main()
