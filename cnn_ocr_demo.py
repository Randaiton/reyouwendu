"""
ONNX + OpenCV DNN 整行数字识别 Demo。

用法：
  python cnn_ocr_demo.py --image screenshots/region0/xxx.jpg --model models/row_crnn.onnx
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

from cnn_ocr_common import decode_ctc_logits, preprocess_row_image, read_image


def resolve_path(path_text: str, project_dir: Path) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return project_dir / path


def parse_row_box(value: str) -> tuple[int, int, int, int]:
    parts = [int(part.strip()) for part in value.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("row-box 格式必须是 x,y,w,h")
    x, y, w, h = parts
    if w <= 0 or h <= 0:
        raise argparse.ArgumentTypeError("row-box 的 w/h 必须大于 0")
    return x, y, w, h


def infer_row_boxes_from_labels(
    labels_path: Path,
    image_shape: tuple[int, int, int],
    project_dir: Path,
) -> list[tuple[int, int, int, int]]:
    """
    从 labels.csv 统计上下两行相对位置，并映射到当前 ROI 图。
    """
    if not labels_path.exists():
        return fallback_row_boxes(image_shape)

    groups: dict[int, list[tuple[float, float, float, float]]] = {}
    with labels_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            source_path = resolve_path(row["source_roi"], project_dir)
            source_image = read_image(source_path)
            if source_image is None:
                continue

            src_h, src_w = source_image.shape[:2]
            if src_h <= 0 or src_w <= 0:
                continue

            row_index = int(row["row_index"])
            x = int(row["x"]) / src_w
            y = int(row["y"]) / src_h
            w = int(row["w"]) / src_w
            h = int(row["h"]) / src_h
            groups.setdefault(row_index, []).append((x, y, w, h))

    if not groups:
        return fallback_row_boxes(image_shape)

    dst_h, dst_w = image_shape[:2]
    boxes: list[tuple[int, int, int, int]] = []
    for row_index in sorted(groups.keys()):
        values = np.asarray(groups[row_index], dtype=np.float32)
        x, y, w, h = np.median(values, axis=0).tolist()
        boxes.append(clip_box(
            round(x * dst_w),
            round(y * dst_h),
            round(w * dst_w),
            round(h * dst_h),
            dst_w,
            dst_h,
        ))
    return boxes


def fallback_row_boxes(image_shape: tuple[int, int, int]) -> list[tuple[int, int, int, int]]:
    """没有 labels.csv 时兜底把 ROI 上下均分成两行。"""
    h, w = image_shape[:2]
    top = clip_box(0, 0, w, round(h * 0.52), w, h)
    bottom = clip_box(0, round(h * 0.45), w, round(h * 0.55), w, h)
    return [top, bottom]


def clip_box(x: int, y: int, w: int, h: int, image_w: int, image_h: int) -> tuple[int, int, int, int]:
    x = max(0, min(x, image_w - 1))
    y = max(0, min(y, image_h - 1))
    w = max(1, min(w, image_w - x))
    h = max(1, min(h, image_h - y))
    return x, y, w, h


def predict_row(net, row_image: np.ndarray) -> str:
    blob = preprocess_row_image(row_image)[np.newaxis, :, :, :]
    net.setInput(blob)
    logits = net.forward()
    return decode_ctc_logits(logits)[0]


def draw_preview(image: np.ndarray, boxes: list[tuple[int, int, int, int]], predictions: list[str]) -> np.ndarray:
    preview = image.copy()
    for index, ((x, y, w, h), prediction) in enumerate(zip(boxes, predictions), start=1):
        cv2.rectangle(preview, (x, y), (x + w, y + h), (0, 255, 255), 1)
        cv2.putText(
            preview,
            f"row{index}: {prediction}",
            (x, max(16, y - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return preview


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ONNX + OpenCV DNN 整行数字识别 Demo")
    parser.add_argument("--image", "-i", required=True,
                        help="待识别 ROI 图片")
    parser.add_argument("--model", "-m", default="models/row_crnn.onnx",
                        help="ONNX 模型路径")
    parser.add_argument("--labels", default="dataset/labels.csv",
                        help="用于统计行位置的 labels.csv")
    parser.add_argument("--row-box", action="append", type=parse_row_box,
                        help="手动指定行框，格式 x,y,w,h；可传多次")
    parser.add_argument("--show", action="store_true",
                        help="显示行框和识别结果")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_dir = Path.cwd()

    image = read_image(args.image)
    if image is None:
        print(f"[ERROR] 图片读取失败: {args.image}")
        return 1

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"[ERROR] ONNX 模型不存在: {model_path}")
        return 1

    boxes = args.row_box or infer_row_boxes_from_labels(Path(args.labels), image.shape, project_dir)
    net = cv2.dnn.readNetFromONNX(str(model_path))

    predictions: list[str] = []
    for index, (x, y, w, h) in enumerate(boxes, start=1):
        crop = image[y:y + h, x:x + w]
        prediction = predict_row(net, crop)
        predictions.append(prediction)
        print(f"第{index}行: {prediction}")

    if args.show:
        preview = draw_preview(image, boxes, predictions)
        cv2.imshow("cnn ocr preview", preview)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    sys.exit(main())
