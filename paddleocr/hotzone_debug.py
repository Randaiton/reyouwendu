from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from tkinter import Tk, filedialog

import cv2

from rec_utils import (
    DEFAULT_CONFIG_PATH,
    get_hotzone,
    get_hotzone_order,
    get_path,
    read_image,
    read_yaml,
    require_box,
    write_image,
    write_yaml,
    draw_hotzone_preview,
)


def choose_image() -> Path | None:
    """
    弹出文件选择框选择参考图片。

    返回:
        Path: 用户选择的图片路径；取消时返回 None。
    """
    root = Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    file_path = filedialog.askopenfilename(
        title="请选择热区调试图片",
        filetypes=[
            ("图片文件", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"),
            ("所有文件", "*.*"),
        ],
    )
    root.destroy()
    return Path(file_path) if file_path else None


def select_hotzones(image, config: dict) -> None:
    """
    逐个框选 8 个热区，并写入配置对象。

    参数:
        image: OpenCV 读取到的整图。
        config: 待更新的 YAML 配置对象。
    """
    window_name = "hotzone_debug - 依次框选 h1 到 h8，Enter/Space 确认"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    for name in get_hotzone_order(config):
        print(f"[INFO] 请框选热区 {name}，Enter/Space 确认，Esc 跳过当前框")
        roi = cv2.selectROI(window_name, image, showCrosshair=True, fromCenter=False)
        x, y, w, h = [int(value) for value in roi]
        hotzone = get_hotzone(config, name)
        if w <= 0 or h <= 0:
            try:
                require_box(hotzone, name)
                print(f"[SKIP] {name} 保留已有坐标")
                continue
            except ValueError:
                cv2.destroyWindow(window_name)
                raise ValueError(f"{name} 未框选，且配置中没有可保留的旧坐标")

        hotzone["box"] = [x, y, w, h]
        hotzone.setdefault("preprocess", {})
        print(f"[SAVE] {name}: {[x, y, w, h]}")

    cv2.destroyWindow(window_name)


def save_preview(image, config: dict, image_path: Path, output_path: Path | None) -> Path:
    """
    保存带热区编号的预览图。

    参数:
        image: 原始整图。
        config: YAML 配置对象。
        image_path: 原始图片路径。
        output_path: 用户指定的预览图路径；为空则使用默认输出目录。

    返回:
        Path: 实际保存的预览图路径。
    """
    if output_path is None:
        output_dir = get_path(config, "outputs") / "previews"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"{image_path.stem}_hotzones_{timestamp}.jpg"

    preview = draw_hotzone_preview(image, config)
    if not write_image(output_path, preview):
        raise RuntimeError(f"预览图保存失败: {output_path}")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PaddleOCR rec 8 热区像素位置调试工具")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH),
                        help="YAML 配置文件路径")
    parser.add_argument("--image",
                        help="参考图片路径；不传则弹出文件选择框")
    parser.add_argument("--preview-only", action="store_true",
                        help="只按现有 YAML 坐标输出预览图，不重新框选")
    parser.add_argument("--output",
                        help="预览图输出路径；不传则保存到 outputs/previews")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = Path(args.config)
    config = read_yaml(config_path)

    image_path = Path(args.image) if args.image else choose_image()
    if image_path is None:
        print("[ERROR] 未选择图片")
        return 1

    image = read_image(image_path)
    if image is None:
        print(f"[ERROR] 图片读取失败: {image_path}")
        return 1

    try:
        if not args.preview_only:
            select_hotzones(image, config)
            write_yaml(config_path, config)
            print(f"[DONE] 热区坐标已保存: {config_path}")

        preview_path = save_preview(
            image,
            config,
            image_path,
            Path(args.output) if args.output else None,
        )
        print(f"[DONE] 预览图已保存: {preview_path}")
    except Exception as exc:
        print(f"[ERROR] {exc}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
