"""
RTSP 摄像头数据流连接 Demo
用于测试视频流 OCR 项目的前期数据采集
摄像头连接信息从 config.yaml 读取
"""
import cv2
import sys
import os
import yaml
import time
import argparse
from datetime import datetime


# ==================== 配置加载 ====================

def load_config(config_path: str) -> dict:
    """从 YAML 文件加载配置"""
    if not os.path.exists(config_path):
        print(f"[ERROR] 配置文件不存在: {config_path}")
        sys.exit(1)
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_rtsp_url(camera: dict) -> str:
    """根据摄像头配置构造 RTSP 地址"""
    host = camera["host"]
    port = camera["port"]
    username = camera.get("username", "")
    password = camera.get("password", "")
    path = camera.get("path", "")

    # 格式: rtsp://[username:password@]host:port/path
    if username and password:
        return f"rtsp://{username}:{password}@{host}:{port}{path}"
    else:
        return f"rtsp://{host}:{port}{path}"


def get_camera(cameras: list, selector: str | int) -> dict | None:
    """根据名称或索引选择摄像头"""
    # 先尝试按索引
    if isinstance(selector, int):
        return cameras[selector]
    # 先尝试按名称匹配
    for cam in cameras:
        if cam["name"] == selector:
            return cam
    # 再尝试按索引数字字符串
    try:
        idx = int(selector)
        return cameras[idx]
    except (ValueError, IndexError):
        pass
    return None


# ==================== RTSP 客户端 ====================

class RTSPClient:
    """RTSP 视频流客户端"""

    def __init__(self, rtsp_url: str, camera_name: str = "",
                 reconnect_delay: float = 3.0, max_retries: int = 10,
                 stream_width: int | None = None, stream_height: int | None = None):
        self.rtsp_url = rtsp_url
        self.camera_name = camera_name
        self.reconnect_delay = reconnect_delay
        self.max_retries = max_retries
        self.stream_width = stream_width      # 请求的视频流宽度
        self.stream_height = stream_height    # 请求的视频流高度
        self.cap: cv2.VideoCapture | None = None
        self.frame_count = 0
        self.start_time = 0.0
        self.retry_count = 0

    def connect(self) -> bool:
        """连接 RTSP 流"""
        self.cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 降低缓冲延迟

        # 设置流分辨率（必须在 open 之后、read 之前设置）
        if self.stream_width is not None:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.stream_width)
        if self.stream_height is not None:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.stream_height)

        if not self.cap.isOpened():
            print(f"[ERROR] 无法连接 RTSP 流: {self.rtsp_url}")
            return False

        # 读取实际生效的分辨率
        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        req_info = ""
        if self.stream_width or self.stream_height:
            req_info = (f", 请求: {self.stream_width or 'auto'}x{self.stream_height or 'auto'}")
        print(f"[INFO] 连接成功 [{self.camera_name}] - "
              f"分辨率: {actual_width}x{actual_height}{req_info}, FPS: {fps:.1f}")
        self.retry_count = 0
        return True

    def reconnect(self) -> bool:
        """断线重连"""
        self.retry_count += 1
        if self.retry_count > self.max_retries:
            print(f"[ERROR] 已达最大重连次数 ({self.max_retries})，放弃重连")
            return False
        self.release()
        print(f"[INFO] 正在重连... ({self.retry_count}/{self.max_retries}, "
              f"{self.reconnect_delay}s 后重试)")
        time.sleep(self.reconnect_delay)
        return self.connect()

    def release(self):
        """释放资源"""
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def read_frame(self) -> tuple[bool, cv2.Mat | None]:
        """读取一帧"""
        if self.cap is None:
            return False, None
        return self.cap.read()

    def run(self, display_opts: dict, capture_opts: dict):
        """
        主循环：读取并显示视频流
        """
        if not self.connect():
            sys.exit(1)

        self.start_time = time.time()
        self.frame_count = 0
        fps_update_interval = 1.0
        last_fps_update = self.start_time
        display_fps = 0.0

        save_dir = capture_opts.get("save_dir")
        save_interval = capture_opts.get("auto_save_interval", 30)
        auto_save_enabled = capture_opts.get("auto_save_enabled", True)
        show_fps = display_opts.get("show_fps", True)
        show_timestamp = display_opts.get("show_timestamp", True)
        window_name = display_opts.get("window_name", "RTSP Stream")
        window_width = display_opts.get("window_width")     # 显示窗口宽度 (null=自适应)
        window_height = display_opts.get("window_height")   # 显示窗口高度

        # 设置显示窗口
        if window_width and window_height:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, window_width, window_height)

        print(f"\n[INFO] 开始读取视频流 [{self.camera_name}], "
              f"按 'q' 退出, 按 's' 手动截图")
        print("=" * 50)

        while True:
            ret, frame = self.read_frame()

            if not ret:
                print("[WARN] 读取帧失败，尝试重连...")
                if not self.reconnect():
                    break
                continue

            self.frame_count += 1

            # --- 帧预处理占位 (后续OCR项目在此扩展) ---
            processed_frame = frame.copy()
            # -------------------------------------------

            # 计算实时FPS
            now = time.time()
            if now - last_fps_update >= fps_update_interval:
                display_fps = self.frame_count / (now - self.start_time + 0.001)
                last_fps_update = now

            # 叠加信息
            if show_fps or show_timestamp:
                self._draw_overlay(processed_frame, display_fps, show_fps, show_timestamp)

            # 显示画面
            cv2.imshow(window_name, processed_frame)

            # 自动保存截图
            if auto_save_enabled and save_dir and save_interval > 0 and self.frame_count % save_interval == 0:
                self._save_screenshot(frame, save_dir, capture_opts.get("image_format", "jpg"))

            # 键盘控制
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("[INFO] 用户退出")
                break
            elif key == ord('s'):
                self._save_screenshot(frame, save_dir or ".",
                                     capture_opts.get("image_format", "jpg"))

        self.release()
        cv2.destroyAllWindows()

        elapsed_total = time.time() - self.start_time
        print(f"\n[INFO] 会话结束 [{self.camera_name}] - "
              f"总帧数: {self.frame_count}, "
              f"平均FPS: {self.frame_count / (elapsed_total + 0.001):.1f}")

    def _draw_overlay(self, frame: cv2.Mat, fps: float,
                      show_fps: bool, show_timestamp: bool):
        """在帧上叠加信息"""
        y = 30
        if self.camera_name:
            cv2.putText(frame, self.camera_name, (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            y += 25
        if show_fps:
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            y += 25
        if show_timestamp:
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(frame, ts, (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            y += 25
        cv2.putText(frame, f"Frame: {self.frame_count}", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    def _save_screenshot(self, frame: cv2.Mat, save_dir: str, fmt: str):
        """保存截图"""
        os.makedirs(save_dir, exist_ok=True)
        filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.{fmt}"
        filepath = os.path.join(save_dir, filename)
        cv2.imwrite(filepath, frame)
        print(f"[SAVE] 截图已保存: {filepath}")


# ==================== 主入口 ====================

def main():
    parser = argparse.ArgumentParser(description="RTSP 摄像头视频流连接测试")

    # 配置文件 & 摄像头选择
    parser.add_argument("--config", "-f", type=str, default="config.yaml",
                        help="配置文件路径, 默认 config.yaml")
    parser.add_argument("--camera", "-c", type=str, default=None,
                        help="摄像头名称或索引 (默认使用第一个 enabled 的摄像头)")

    # 直接指定URL时跳过配置文件中的摄像头列表
    parser.add_argument("--url", "-u", type=str, default=None,
                        help="直接指定 RTSP 地址 (覆盖配置文件)")

    # 显示 & 采集参数 (可覆盖配置文件中的值)
    parser.add_argument("--no-display", action="store_true",
                        help="不显示画面")
    parser.add_argument("--save-dir", "-d", type=str, default=None,
                        help="截图保存目录 (覆盖配置文件)")
    parser.add_argument("--save-interval", "-i", type=int, default=None,
                        help="自动截图间隔/帧数 (覆盖配置文件)")
    parser.add_argument("--auto-capture", dest="auto_capture", action="store_true",
                        default=None,
                        help="启用自动截图 (覆盖配置文件)")
    parser.add_argument("--no-auto-capture", dest="auto_capture", action="store_false",
                        help="关闭自动截图 (覆盖配置文件)")

    args = parser.parse_args()

    # --- 加载配置文件 ---
    config_path = args.config
    # 支持相对于脚本所在目录的路径
    if not os.path.isabs(config_path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, config_path)
    cfg = load_config(config_path)

    # --- 确定 RTSP 地址 ---
    camera = None  # 使用命令行URL时为None
    if args.url:
        # 命令行直接指定URL，最高优先级
        rtsp_url = args.url
        camera_name = "命令行指定"
        reconnect_cfg = cfg.get("reconnect", {})
    else:
        cameras = cfg.get("cameras", [])
        if not cameras:
            print("[ERROR] 配置文件中没有摄像头信息")
            sys.exit(1)

        # 选择摄像头
        if args.camera is not None:
            camera = get_camera(cameras, args.camera)
            if camera is None:
                print(f"[ERROR] 未找到摄像头: {args.camera}")
                print(f"[INFO] 可用摄像头: "
                      f"{[c['name'] for c in cameras]}")
                sys.exit(1)
        else:
            # 默认选第一个 enabled 的
            enabled = [c for c in cameras if c.get("enabled", True)]
            camera = enabled[0] if enabled else cameras[0]

        rtsp_url = build_rtsp_url(camera)
        camera_name = camera.get("name", "unknown")
        reconnect_cfg = camera.get("reconnect", cfg.get("reconnect", {}))

    # --- 显示 & 采集配置 ---
    display_opts = cfg.get("display", {})
    capture_opts = cfg.get("capture", {})

    # 命令行参数覆盖配置文件
    if args.no_display:
        display_opts["enabled"] = False
    if args.save_dir is not None:
        capture_opts["save_dir"] = args.save_dir
    if args.save_interval is not None:
        capture_opts["auto_save_interval"] = args.save_interval
    if args.auto_capture is not None:
        capture_opts["auto_save_enabled"] = args.auto_capture

    # --- 启动客户端 ---
    # 视频流分辨率 (从摄像头配置读取, 命令行URL模式不设置)
    stream_width = camera.get("width") if camera else None
    stream_height = camera.get("height") if camera else None

    client = RTSPClient(
        rtsp_url=rtsp_url,
        camera_name=camera_name,
        reconnect_delay=reconnect_cfg.get("delay_seconds", 3.0),
        max_retries=reconnect_cfg.get("max_retries", 10),
        stream_width=stream_width,
        stream_height=stream_height,
    )
    client.run(display_opts=display_opts, capture_opts=capture_opts)


if __name__ == "__main__":
    main()
