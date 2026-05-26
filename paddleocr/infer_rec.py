from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from tkinter import Tk, filedialog
from typing import Any

from rec_utils import (
    DEFAULT_CONFIG_PATH,
    apply_preprocess,
    crop_hotzone,
    draw_hotzone_preview,
    get_hotzone,
    get_hotzone_order,
    get_path,
    is_valid_prediction,
    path_for_config,
    read_image,
    read_yaml,
    require_box,
    write_image,
)


def choose_image() -> Path | None:
    """
    弹出文件选择框选择推理图片。

    返回:
        Path: 用户选择的图片路径；取消时返回 None。
    """
    root = Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    file_path = filedialog.askopenfilename(
        title="请选择推理测试图片",
        filetypes=[
            ("图片文件", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"),
            ("所有文件", "*.*"),
        ],
    )
    root.destroy()
    return Path(file_path) if file_path else None


def extract_result(result: Any) -> tuple[str, float | None, dict[str, Any]]:
    """
    从 PaddleOCR Result 对象中提取识别文本和置信度。

    参数:
        result: PaddleOCR TextRecognition 返回的 Result 对象。

    返回:
        tuple: (识别文本, 置信度, 原始 JSON 字典)。
    """
    data = getattr(result, "json", None)
    if callable(data):
        data = data()
    if not isinstance(data, dict):
        data = {}

    payload = data.get("res", data)
    text = str(payload.get("rec_text", payload.get("text", "")))
    score = payload.get("rec_score", payload.get("score"))
    if score is not None:
        score = float(score)
    return text, score, data


def crop_for_infer(config: dict[str, Any], image_path: Path, output_dir: Path) -> list[dict[str, Any]]:
    """
    按 YAML 热区裁剪推理图片。

    参数:
        config: YAML 配置对象。
        image_path: 原始推理图片路径。
        output_dir: 当前图片的输出目录。

    返回:
        list[dict]: 每个热区的裁剪信息。
    """
    image = read_image(image_path)
    if image is None:
        raise RuntimeError(f"图片读取失败: {image_path}")

    crops_dir = output_dir / "crops"
    crops: list[dict[str, Any]] = []
    for name in get_hotzone_order(config):
        hotzone = get_hotzone(config, name)
        box = require_box(hotzone, name)
        crop = crop_hotzone(image, box)
        crop = apply_preprocess(crop, hotzone.get("preprocess"))
        crop_path = crops_dir / f"{name}.png"
        if not write_image(crop_path, crop):
            raise RuntimeError(f"推理裁剪图保存失败: {crop_path}")
        crops.append({
            "name": name,
            "box": box,
            "crop_path": crop_path,
        })
    return crops


def run_recognition(config: dict[str, Any], model_dir: Path, crop_paths: list[Path], batch_size: int) -> list[Any]:
    """
    调用 PaddleOCR TextRecognition 进行 CPU 推理。

    参数:
        config: YAML 配置对象。
        model_dir: 导出的静态推理模型目录。
        crop_paths: 8 个热区裁剪图路径。
        batch_size: 推理 batch size。

    返回:
        list: PaddleOCR Result 对象列表。
    """
    from paddleocr import TextRecognition

    model = TextRecognition(
        model_name=config["model"]["name"],
        model_dir=path_for_config(model_dir),
        device=config["model"].get("device", "cpu"),
        engine=config["model"].get("engine", "paddle_static"),
        cpu_threads=int(config["model"].get("cpu_threads", 8)),
    )
    return list(model.predict(input=[path_for_config(path) for path in crop_paths], batch_size=batch_size))


def write_json(path: Path, data: dict[str, Any]) -> None:
    """
    写入 JSON 文件，保留中文字符为直接可读文本。

    参数:
        path: JSON 文件路径。
        data: 写入内容。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PaddleOCR-v5 mobile rec 8 热区 CPU 推理测试")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH),
                        help="YAML 配置文件路径")
    parser.add_argument("--image",
                        help="推理图片路径；不传则弹出文件选择框")
    parser.add_argument("--model-dir",
                        help="导出的 rec inference 模型目录；不传则读取配置 paths.models/model.inference_dir_name")
    parser.add_argument("--output-dir",
                        help="推理输出目录；不传则保存到 outputs/infer")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="TextRecognition 推理 batch size")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = read_yaml(args.config)

    image_path = Path(args.image) if args.image else choose_image()
    if image_path is None:
        print("[ERROR] 未选择图片")
        return 1

    model_dir = Path(args.model_dir) if args.model_dir else (
        get_path(config, "models") / config["model"]["inference_dir_name"]
    )
    if not model_dir.exists():
        print(f"[ERROR] 推理模型目录不存在: {model_dir}")
        return 1

    base_output_dir = Path(args.output_dir) if args.output_dir else get_path(config, "outputs") / "infer"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = base_output_dir / f"{image_path.stem}_{timestamp}"

    try:
        crops = crop_for_infer(config, image_path, output_dir)
        results = run_recognition(
            config,
            model_dir,
            [item["crop_path"] for item in crops],
            batch_size=args.batch_size,
        )

        records = []
        preview_labels: dict[str, str] = {}
        for crop, result in zip(crops, results):
            text, score, raw = extract_result(result)
            valid = is_valid_prediction(text)
            records.append({
                "hotzone": crop["name"],
                "box": list(crop["box"]),
                "crop_path": path_for_config(crop["crop_path"]),
                "text": text,
                "score": score,
                "valid": valid,
                "status": "ok" if valid else "invalid_format",
                "raw": raw,
            })
            preview_labels[crop["name"]] = f"{crop['name']}:{text}"

        source_image = read_image(image_path)
        preview = draw_hotzone_preview(source_image, config, labels=preview_labels)
        preview_path = output_dir / "preview.jpg"
        if not write_image(preview_path, preview):
            raise RuntimeError(f"推理预览图保存失败: {preview_path}")

        json_path = output_dir / "result.json"
        write_json(json_path, {
            "image_path": path_for_config(image_path),
            "model_dir": path_for_config(model_dir),
            "results": records,
        })

        for record in records:
            print(f"{record['hotzone']}: {record['text']} score={record['score']} status={record['status']}")
        print(f"[DONE] 推理结果已保存: {json_path}")
        print(f"[DONE] 推理预览图已保存: {preview_path}")
    except Exception as exc:
        print(f"[ERROR] {exc}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
