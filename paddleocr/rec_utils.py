from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = PROJECT_DIR / "config" / "rec_finetune.yml"
LABEL_MARKER = "__labels_"
LABEL_PATTERN = re.compile(r"^\d{2,3}$")


def read_yaml(path: str | Path) -> dict[str, Any]:
    """
    读取 YAML 配置文件。

    参数:
        path: YAML 文件路径。

    返回:
        dict: 配置内容；空文件返回空字典。
    """
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def write_yaml(path: str | Path, data: dict[str, Any]) -> None:
    """
    写入 YAML 配置文件，保留中文字符为直接可读文本。

    参数:
        path: YAML 文件路径。
        data: 需要写入的配置字典。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


def resolve_project_path(path_text: str | Path, project_dir: Path = PROJECT_DIR) -> Path:
    """
    将配置中的路径解析为绝对路径。

    参数:
        path_text: 配置中的路径，支持绝对路径和相对 paddleocr 目录的路径。
        project_dir: 工程根目录，默认是当前 paddleocr 目录。

    返回:
        Path: 解析后的绝对路径。
    """
    path = Path(path_text)
    if path.is_absolute():
        return path
    return project_dir / path


def path_for_config(path: Path) -> str:
    """
    将路径转换为 PaddleOCR 配置更稳定的正斜杠格式。

    参数:
        path: 需要写入配置或命令的路径。

    返回:
        str: 正斜杠路径字符串。
    """
    return path.resolve().as_posix()


def get_path(config: dict[str, Any], key: str) -> Path:
    """
    读取 paths 配置项并解析为绝对路径。

    参数:
        config: 总配置。
        key: paths 下的键名。

    返回:
        Path: 解析后的绝对路径。
    """
    return resolve_project_path(config["paths"][key])


def list_image_files(path: str | Path, extensions: list[str]) -> list[Path]:
    """
    列出图片文件，目录会递归扫描。

    参数:
        path: 图片文件或目录。
        extensions: 支持的图片后缀列表。

    返回:
        list[Path]: 排序后的图片路径列表。
    """
    source = Path(path)
    suffixes = {item.lower() for item in extensions}
    if source.is_file():
        return [source] if source.suffix.lower() in suffixes else []
    if not source.exists():
        return []
    return sorted(
        item for item in source.rglob("*")
        if item.is_file() and item.suffix.lower() in suffixes
    )


def read_image(image_path: str | Path) -> np.ndarray | None:
    """
    读取图片，兼容包含中文的 Windows 路径。

    参数:
        image_path: 图片文件路径。

    返回:
        numpy.ndarray: OpenCV BGR 图片；读取失败返回 None。
    """
    image_path = Path(image_path)
    data = np.fromfile(str(image_path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def write_image(image_path: str | Path, image: np.ndarray) -> bool:
    """
    保存图片，兼容包含中文的 Windows 路径。

    参数:
        image_path: 图片保存路径。
        image: OpenCV 图片数组。

    返回:
        bool: 保存成功返回 True，否则返回 False。
    """
    image_path = Path(image_path)
    image_path.parent.mkdir(parents=True, exist_ok=True)
    success, encoded = cv2.imencode(image_path.suffix, image)
    if not success:
        return False
    encoded.tofile(str(image_path))
    return True


def get_hotzone_order(config: dict[str, Any]) -> list[str]:
    """
    获取热区顺序。

    参数:
        config: 总配置。

    返回:
        list[str]: 热区名称列表。
    """
    order = list(config.get("hotzone_order") or [])
    if len(order) != 8:
        raise ValueError("hotzone_order 必须包含 8 个热区名称")
    return order


def get_hotzone(config: dict[str, Any], name: str) -> dict[str, Any]:
    """
    获取单个热区配置。

    参数:
        config: 总配置。
        name: 热区名称。

    返回:
        dict: 热区配置。
    """
    hotzones = config.get("hotzones") or {}
    if name not in hotzones:
        raise ValueError(f"配置中缺少热区: {name}")
    return hotzones[name]


def require_box(hotzone: dict[str, Any], name: str) -> tuple[int, int, int, int]:
    """
    读取并校验热区像素框。

    参数:
        hotzone: 单个热区配置。
        name: 热区名称，用于错误提示。

    返回:
        tuple[int, int, int, int]: (x, y, w, h)。
    """
    box = hotzone.get("box")
    if not isinstance(box, list) or len(box) != 4:
        raise ValueError(f"{name} 未配置有效 box，请先运行 hotzone_debug.py 框选热区")
    x, y, w, h = [int(value) for value in box]
    if w <= 0 or h <= 0:
        raise ValueError(f"{name} 的 box 宽高必须大于 0: {box}")
    return x, y, w, h


def clip_box(box: tuple[int, int, int, int], image_shape: tuple[int, ...]) -> tuple[int, int, int, int]:
    """
    将热区框限制在图片范围内。

    参数:
        box: 热区框 (x, y, w, h)。
        image_shape: 图片 shape。

    返回:
        tuple[int, int, int, int]: 裁剪后的热区框。
    """
    x, y, w, h = box
    image_h, image_w = image_shape[:2]
    x = max(0, min(int(x), image_w - 1))
    y = max(0, min(int(y), image_h - 1))
    w = max(1, min(int(w), image_w - x))
    h = max(1, min(int(h), image_h - y))
    return x, y, w, h


def crop_hotzone(image: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    """
    根据热区框裁剪图片。

    参数:
        image: 原始整图。
        box: 热区框 (x, y, w, h)。

    返回:
        numpy.ndarray: 热区裁剪图。
    """
    x, y, w, h = clip_box(box, image.shape)
    return image[y:y + h, x:x + w].copy()


def apply_preprocess(image: np.ndarray, preprocess: dict[str, Any] | None) -> np.ndarray:
    """
    应用单个热区的预处理配置。

    参数:
        image: 热区裁剪图。
        preprocess: 预处理配置，支持 gray、contrast、brightness、sharpen_strength、threshold。

    返回:
        numpy.ndarray: 预处理后的 BGR 图片。
    """
    preprocess = preprocess or {}
    result = image.copy()

    contrast = float(preprocess.get("contrast", 1.0))
    brightness = float(preprocess.get("brightness", 0))
    if contrast != 1.0 or brightness != 0:
        result = cv2.convertScaleAbs(result, alpha=contrast, beta=brightness)

    if preprocess.get("gray", False):
        result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

    sharpen_strength = float(preprocess.get("sharpen_strength", 0.0))
    if sharpen_strength > 0:
        blur = cv2.GaussianBlur(result, (0, 0), sigmaX=1.0)
        result = cv2.addWeighted(result, 1.0 + sharpen_strength, blur, -sharpen_strength, 0)

    threshold = preprocess.get("threshold") or {}
    if threshold.get("enabled", False):
        gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        value = int(threshold.get("value", 0))
        max_value = int(threshold.get("max_value", 255))
        mode_text = str(threshold.get("mode", "binary")).lower()
        threshold_mode = cv2.THRESH_BINARY_INV if mode_text == "binary_inv" else cv2.THRESH_BINARY
        _, mask = cv2.threshold(gray, value, max_value, threshold_mode)
        result = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    return result


def parse_labels_from_filename(image_path: str | Path, expected_count: int = 8) -> list[str]:
    """
    从文件名中解析 8 个数字标签。

    参数:
        image_path: 原始训练图片路径。
        expected_count: 需要解析的标签数量，默认 8 个。

    返回:
        list[str]: 标签列表。
    """
    stem = Path(image_path).stem
    if LABEL_MARKER not in stem:
        raise ValueError(f"文件名缺少 {LABEL_MARKER}: {Path(image_path).name}")

    labels_text = stem.split(LABEL_MARKER, 1)[1]
    labels = labels_text.split("_")
    if len(labels) != expected_count:
        raise ValueError(f"文件名标签数量必须为 {expected_count} 个: {Path(image_path).name}")
    for label in labels:
        if not LABEL_PATTERN.match(label):
            raise ValueError(f"标签必须是 2-3 位数字: {label}")
    return labels


def make_dataset_image_name(source_path: Path, hotzone_name: str) -> str:
    """
    生成裁剪图片文件名。

    参数:
        source_path: 原始图片路径。
        hotzone_name: 热区名称。

    返回:
        str: 数据集裁剪图文件名。
    """
    return f"{source_path.stem}__{hotzone_name}.png"


def draw_hotzone_preview(
    image: np.ndarray,
    config: dict[str, Any],
    labels: dict[str, str] | None = None,
) -> np.ndarray:
    """
    绘制热区预览图。

    参数:
        image: 原始整图。
        config: 总配置。
        labels: 可选热区显示文本。

    返回:
        numpy.ndarray: 带热区框的预览图。
    """
    preview = image.copy()
    labels = labels or {}
    colors = [
        (0, 255, 255),
        (0, 200, 0),
        (255, 180, 0),
        (255, 0, 255),
        (0, 128, 255),
        (255, 255, 0),
        (128, 255, 0),
        (255, 128, 128),
    ]

    for index, name in enumerate(get_hotzone_order(config)):
        hotzone = get_hotzone(config, name)
        x, y, w, h = clip_box(require_box(hotzone, name), image.shape)
        color = colors[index % len(colors)]
        text = labels.get(name, name)
        cv2.rectangle(preview, (x, y), (x + w, y + h), color, 2)
        cv2.putText(
            preview,
            text,
            (x, max(18, y - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )
    return preview


def is_valid_prediction(text: str) -> bool:
    """
    校验推理结果是否为 2-3 位数字。

    参数:
        text: 识别文本。

    返回:
        bool: 符合格式返回 True。
    """
    return bool(LABEL_PATTERN.match(text))
