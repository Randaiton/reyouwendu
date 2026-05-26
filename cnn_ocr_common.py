"""
CNN/CRNN 整行数字识别的公共工具。

训练脚本和 ONNX 推理脚本必须共用这里的字符表、图像尺寸、预处理和 CTC 解码，
避免训练与部署阶段行为不一致。
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


DIGITS = "0123456789"
BLANK_INDEX = len(DIGITS)
NUM_CLASSES = len(DIGITS) + 1
IMAGE_HEIGHT = 48
IMAGE_WIDTH = 160

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def read_image(path: str | Path) -> np.ndarray | None:
    """读取图片，支持包含中文的 Windows 路径。"""
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def write_image(path: str | Path, image: np.ndarray) -> bool:
    """写入图片，支持包含中文的 Windows 路径。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix or ".png"
    ok, encoded = cv2.imencode(ext, image)
    if not ok:
        return False
    encoded.tofile(str(path))
    return True


def list_image_files(input_path: str | Path) -> list[Path]:
    """递归列出图片文件。"""
    root = Path(input_path)
    if root.is_file():
        return [root] if root.suffix.lower() in IMAGE_EXTENSIONS else []

    files: list[Path] = []
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            files.append(path)
    return sorted(files)


def normalize_label(label: str) -> str:
    """清洗标签，只接受 2-3 位数字。"""
    value = label.strip()
    if not value.isdigit() or len(value) not in (2, 3):
        raise ValueError("标签必须是 2-3 位数字，例如 23、223、300")
    return value


def encode_label(label: str) -> list[int]:
    """将数字字符串编码为 CTC 训练目标。"""
    value = normalize_label(label)
    return [DIGITS.index(char) for char in value]


def preprocess_row_image(
    image: np.ndarray,
    height: int = IMAGE_HEIGHT,
    width: int = IMAGE_WIDTH,
) -> np.ndarray:
    """
    将整行数字图预处理为 CRNN 输入。

    输出 shape 为 (1, height, width)，数值范围约为 [-1, 1]。
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    h, w = gray.shape[:2]
    if h <= 0 or w <= 0:
        raise ValueError("输入图片尺寸无效")

    scale = min(height / h, width / w)
    resized_w = max(1, min(width, int(round(w * scale))))
    resized_h = max(1, min(height, int(round(h * scale))))
    resized = cv2.resize(gray, (resized_w, resized_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((height, width), dtype=np.uint8)
    y = (height - resized_h) // 2
    x = 0
    canvas[y:y + resized_h, x:x + resized_w] = resized

    normalized = canvas.astype(np.float32) / 255.0
    normalized = (normalized - 0.5) / 0.5
    return normalized[np.newaxis, :, :]


def decode_ctc_logits(logits: np.ndarray) -> list[str]:
    """
    CTC greedy decode。

    支持 ONNX 输出 shape:
      - (N, T, C)
      - (T, N, C)
      - (T, C)
    """
    array = np.asarray(logits)
    if array.ndim == 2:
        array = array[np.newaxis, :, :]
    elif array.ndim == 3 and array.shape[0] > array.shape[1] and array.shape[1] == 1:
        array = np.transpose(array, (1, 0, 2))

    indices = np.argmax(array, axis=-1)
    results: list[str] = []
    for sequence in indices:
        chars: list[str] = []
        previous = BLANK_INDEX
        for index in sequence.tolist():
            if index != BLANK_INDEX and index != previous:
                chars.append(DIGITS[index])
            previous = index
        results.append("".join(chars))
    return results


def relative_path(path: str | Path, base: str | Path) -> str:
    """返回相对路径；如果无法相对化，则返回原路径字符串。"""
    path = Path(path)
    base = Path(base)
    try:
        return str(path.resolve().relative_to(base.resolve()))
    except ValueError:
        return str(path)
