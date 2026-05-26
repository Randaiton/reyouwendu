"""
七段数码管识别 Demo

用途：
1. 手动选择一张或多张 ROI 图片。
2. 每张 ROI 中自动分离上下两行数字。
3. 每行按七段数码管规则识别 2-3 位数字。

说明：
这是前期验证用 demo，阈值参数保留在命令行中，方便根据实际截图微调。

核心算法流程：
  加载图片 → 放大 → HSV提取亮笔画 → 形态学去噪
  → 水平投影找行 → 垂直投影切分每位数字
  → 每位数字7段区域采样 → 与标准段码匹配 → 输出结果
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

# ============================================================
# 七段数码管段码映射表
# ============================================================
# 七段数码管由 a~g 共 7 个 LED 段组成，排列如下:
#
#         aaaa
#       f    b
#       f    b
#         gggg
#       e    c
#       e    c
#         dddd
#
# 每个数字通过点亮不同段的组合来显示，例如数字 "1" 只点亮 b、c 两段。

# 数字 → 点亮段集合 (a~g)
SEGMENT_MAP = {
    "0": "abcdef",   # 0: a b c d e f    (仅 g 不亮)
    "1": "bc",       # 1:     b c
    "2": "abged",    # 2: a b   g e d
    "3": "abgcd",    # 3: a b   g c d
    "4": "fgbc",     # 4:   f g b c
    "5": "afgcd",    # 5: a f g   c d
    "6": "afgecd",   # 6: a f g e c d    (仅 b 不亮)
    "7": "abc",      # 7: a   b c
    "8": "abcdefg",  # 8: a b c d e f g  (全亮)
    "9": "abfgcd",   # 9: a b f g c d    (仅 e 不亮)
}

# 标准段顺序，遍历时使用
SEGMENT_ORDER = ("a", "b", "c", "d", "e", "f", "g")


# ============================================================
# 配置与结果数据结构
# ============================================================

@dataclass
class DetectConfig:
    """识别参数配置"""
    scale: float                # 图片放大倍数（小ROI需要放大才能准确识别）
    min_v: int                  # HSV 明度(V)下限，低于此值的像素视为背景
    min_s: int                  # HSV 饱和度(S)下限，用于保留彩色数码管笔画
    min_chroma: int             # BGR 最大/最小通道差值下限，用于排除灰底高光
    color_percentile: float     # 在候选颜色内只保留更强的发光像素，抑制内部泛光
    white_v: int                # 白色/浅色笔画的明度下限（白色笔画饱和度低，需单独判断）
    color_mode: str             # 笔画颜色模式：led/red/green/any
    include_white: bool         # 是否额外保留低饱和高亮白色笔画
    row_threshold_ratio: float  # 行投影阈值比例：投影峰值 * ratio = 行分割线
    col_threshold_ratio: float  # 列投影阈值比例：投影峰值 * ratio = 数字间分界
    expected_digits: int | None # 指定每行数字位数(2/3)，None则自动检测
    show: bool                  # 是否显示预览窗口
    debug: bool                 # 是否打印每个数字的段占比和匹配距离


@dataclass
class RowResult:
    """一行数字的识别结果"""
    number: str                                     # 识别出的数字字符串（如 "23"）
    row_box: tuple[int, int, int, int]              # 该行在图片中的包围框 (x1, y1, x2, y2)
    digit_boxes: list[tuple[int, int, int, int]]    # 每个数字的包围框列表
    digit_scores: list[dict[str, float]]            # 每个数字的7段匹配分数


# ============================================================
# 图像读取
# ============================================================

def read_image(path: str) -> np.ndarray | None:
    """
    读取图片，支持包含中文的 Windows 路径。
    OpenCV 的 imread 不支持中文路径，这里用 np.fromfile + imdecode 绕过。
    """
    data = np.fromfile(path, dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def choose_images() -> list[str]:
    """弹出 Windows 文件选择框，手动选择一张或多张图片。"""
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception as exc:
        print(f"[ERROR] 无法打开文件选择框: {exc}")
        return []

    root = tk.Tk()
    root.withdraw()                         # 隐藏主窗口
    root.attributes("-topmost", True)       # 文件选择框置顶
    paths = filedialog.askopenfilenames(
        title="选择 ROI 图片",
        filetypes=[
            ("图片文件", "*.jpg;*.jpeg;*.png;*.bmp"),
            ("所有文件", "*.*"),
        ],
    )
    root.destroy()
    return list(paths)


# ============================================================
# 笔画提取（HSV阈值 + 形态学处理）
# ============================================================

def keep_strong_color(candidate: np.ndarray, color_score: np.ndarray, cfg: DetectConfig) -> np.ndarray:
    """
    在候选颜色区域中只保留更强的发光像素。
    复杂光照下，数字内部镂空区域经常有弱泛光；按分位数过滤可以把弱泛光压掉。
    """
    values = color_score[candidate]
    if values.size == 0:
        return candidate

    threshold = np.percentile(values, cfg.color_percentile)
    return candidate & (color_score >= threshold)


def build_led_mask(image: np.ndarray, cfg: DetectConfig, color_mode: str | None = None) -> np.ndarray:
    """
    从 BGR 图片中提取亮起的数码管笔画，返回二值 mask。

    策略：
      1. 转 HSV 色彩空间。
      2. 彩色数码管：明度(V)、饱和度(S)、BGR色度差都高的像素视为候选笔画。
      3. 默认只保留红/粉、黄绿两个色相范围，排除复杂光照下的灰底高光。
      4. 白色数码管/低饱和浅色：仅在 include_white=True 时额外启用。
      5. 形态学开运算去除孤立噪点，闭运算连接断裂的笔画段。
    """
    mode = color_mode or cfg.color_mode
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue, sat, val = cv2.split(hsv)  # 分别取 H, S, V 三通道
    bgr_max = image.max(axis=2)
    bgr_min = image.min(axis=2)
    chroma = bgr_max - bgr_min

    # 彩色笔画：V高 + S高 + 色度差高。第三个条件用于过滤亮灰色背景和反光。
    colorful_led = (val >= cfg.min_v) & (sat >= cfg.min_s) & (chroma >= cfg.min_chroma)
    color_score = sat.astype(np.float32) * val.astype(np.float32)

    # OpenCV HSV 的 H 范围是 0~179。
    # 红/粉：靠近 0 或 145~179；黄绿/绿色：约 25~95。
    red_led = keep_strong_color(colorful_led & ((hue <= 12) | (hue >= 145)), color_score, cfg)
    green_led = keep_strong_color(colorful_led & (hue >= 25) & (hue <= 95), color_score, cfg)

    if mode == "red":
        colored_led = red_led
    elif mode == "green":
        colored_led = green_led
    elif mode == "any":
        colored_led = keep_strong_color(colorful_led, color_score, cfg)
    else:
        colored_led = red_led | green_led

    # 白色/浅色笔画默认不启用，否则复杂光照下高亮灰底会被整片选中。
    white_led = (val >= cfg.white_v) if cfg.include_white else False
    # 合并笔画
    mask = np.where(colored_led | white_led, 255, 0).astype(np.uint8)

    h, w = mask.shape[:2]

    # --- 形态学去噪 ---
    # 开运算（先腐蚀后膨胀）：消除孤立的小噪点
    open_size = max(1, round(min(h, w) / 100))
    # 闭运算（先膨胀后腐蚀）：连接同一数字内断裂的笔画段
    close_w = max(2, round(w / 60))
    close_h = max(2, round(h / 60))

    if open_size > 1:
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_RECT, (open_size, open_size)),
        )
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (close_w, close_h)),
    )
    return mask


# ============================================================
# 辅助函数：区间合并与激活区间查找
# ============================================================

def merge_ranges(
    ranges: list[tuple[int, int]],
    max_gap: int,
) -> list[tuple[int, int]]:
    """
    合并间隔不超过 max_gap 的连续区间。
    例如 ranges=[(0,10),(15,20)], max_gap=8 → 合并为 [(0,20)]
            ranges=[(0,10),(15,20)], max_gap=3 → 保持 [(0,10),(15,20)]
    """
    if not ranges:
        return []

    merged = [ranges[0]]
    for start, end in ranges[1:]:
        prev_start, prev_end = merged[-1]
        if start - prev_end <= max_gap:
            # 当前区间与上一个区间足够近，合并
            merged[-1] = (prev_start, end)
        else:
            merged.append((start, end))
    return merged


def active_ranges(active: np.ndarray) -> list[tuple[int, int]]:
    """
    在 bool 数组中找到所有连续为 True 的区间。
    例如 input=[0,1,1,0,0,1,0] → [(1,3), (5,6)]
    用于从投影曲线中提取连续有笔画的区域。
    """
    ranges: list[tuple[int, int]] = []
    start: int | None = None

    for index, is_active in enumerate(active):
        if is_active and start is None:
            start = index           # 进入激活区域
        elif not is_active and start is not None:
            ranges.append((start, index))  # 离开激活区域
            start = None

    if start is not None:
        ranges.append((start, len(active)))  # 处理末尾未闭合的区间
    return ranges


# ============================================================
# 行定位：水平投影分析
# ============================================================

def find_rows(mask: np.ndarray, cfg: DetectConfig) -> list[tuple[int, int, int, int]]:
    """
    在二值 mask 中定位数字行的位置。

    算法：
      1. 水平投影：统计每一行的白色像素数。
      2. 平滑投影曲线（卷积均值滤波），减少笔画间隙造成的断裂。
      3. 用投影峰值 * row_threshold_ratio 作为阈值，提取有笔画的 y 区间。
      4. 合并间距很近的区间。
      5. 过滤太矮的区间，并在每行内找到 x 范围。
      6. 如果超过 2 行，保留白色像素最多的 2 行（上下两行）。
      7. 按 y 坐标从小到大返回（先上行后下行）。
    """
    h, w = mask.shape[:2]

    # 水平投影：统计每行白色像素数
    projection = np.count_nonzero(mask, axis=1).astype(np.float32)

    # 卷积平滑投影曲线，消除笔画间缝隙造成的短暂跌落
    smooth_kernel = max(3, round(h / 25))
    projection = np.convolve(
        projection,
        np.ones(smooth_kernel, dtype=np.float32) / smooth_kernel,
        mode="same",
    )

    # 动态阈值 = 投影最大值 * 比例系数（系数越低越容易检测到暗行）
    threshold = max(1.0, projection.max() * cfg.row_threshold_ratio)
    # 找到投影高于阈值的连续区间
    rows = active_ranges(projection >= threshold)
    # 合并间距小的行（同一行数字内可能因笔画间隙被切开）
    rows = merge_ranges(rows, max_gap=max(2, round(h * 0.08)))

    # 过滤太矮的行（可能为噪点）
    min_row_h = max(4, round(h * 0.12))
    candidates: list[tuple[int, int, int, int]] = []
    for y1, y2 in rows:
        if y2 - y1 < min_row_h:
            continue
        # 在该行内找有笔画的 x 范围
        row_mask = mask[y1:y2, :]
        cols = np.where(np.count_nonzero(row_mask, axis=0) > 0)[0]
        if cols.size == 0:
            continue
        x1, x2 = int(cols[0]), int(cols[-1] + 1)
        candidates.append((x1, y1, x2, y2))

    # 只保留像素数最多的 2 行（假设只有上下两行数字）
    if len(candidates) > 2:
        candidates = sorted(
            candidates,
            key=lambda box: np.count_nonzero(mask[box[1]:box[3], box[0]:box[2]]),
            reverse=True,
        )[:2]

    # 按 y 坐标排序：上方行在前
    return sorted(candidates, key=lambda box: box[1])


# ============================================================
# 数字分割：垂直投影分析
# ============================================================

def split_digits(
    row_mask: np.ndarray,
    row_origin: tuple[int, int],
    cfg: DetectConfig,
) -> list[tuple[int, int, int, int]]:
    """
    在一行的 mask 中切分出每个数字。

    算法：
      1. 水平方向膨胀，将同一数字内有微小断开的笔画连为一体。
      2. 垂直投影：统计每列的白色像素数。
      3. 用投影峰值 * col_threshold_ratio 作为阈值，提取有笔画的 x 区间。
      4. 在每列的区间内找 y 范围，得到每个数字的包围框。
      5. 如果指定了 expected_digits 但自动检测数量不匹配：
         - 数量不够 → 用等宽均分法补位
         - 数量超出 → 保留像素最多的 N 个
      6. 返回的坐标转换为全图绝对坐标。
    """
    h, w = row_mask.shape[:2]

    # 水平方向膨胀，连接同一数字内可能存在的微小笔画断裂
    kernel_w = max(2, round(w / 35))
    merged_mask = cv2.dilate(
        row_mask,
        cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, 1)),
        iterations=1,
    )

    # 垂直投影：统计每列白色像素数
    projection = np.count_nonzero(merged_mask, axis=0).astype(np.float32)
    # 动态阈值
    threshold = max(1.0, projection.max() * cfg.col_threshold_ratio)
    columns = active_ranges(projection >= threshold)
    # 合并间隙很小的列区间（数字内部数字间距）
    columns = merge_ranges(columns, max_gap=max(1, round(w * 0.03)))

    # 过滤太窄的区间
    min_digit_w = max(4, round(w * 0.10))
    candidates: list[tuple[int, int, int, int]] = []
    for x1, x2 in columns:
        if x2 - x1 < min_digit_w:
            continue
        # 在该列区间内找有笔画的 y 范围
        digit_mask = row_mask[:, x1:x2]
        ys = np.where(np.count_nonzero(digit_mask, axis=1) > 0)[0]
        if ys.size == 0:
            continue
        y1, y2 = int(ys[0]), int(ys[-1] + 1)
        candidates.append((x1, y1, x2, y2))

    target_digits = cfg.expected_digits or infer_digit_count(row_mask, candidates)

    # --- 校正数字个数 ---
    if target_digits and len(candidates) != target_digits:
        # 数量不够时用等宽均分法补齐
        candidates = split_digits_by_width(row_mask, target_digits)

    if target_digits and len(candidates) > target_digits:
        # 数量超出时保留白色像素最多的
        candidates = sorted(
            candidates,
            key=lambda box: np.count_nonzero(row_mask[box[1]:box[3], box[0]:box[2]]),
            reverse=True,
        )[:target_digits]

    # 转换为原图绝对坐标
    ox, oy = row_origin
    absolute = [(x1 + ox, y1 + oy, x2 + ox, y2 + oy) for x1, y1, x2, y2 in candidates]
    # 按 x 坐标排序：左侧数字在前
    return sorted(absolute, key=lambda box: box[0])


def infer_digit_count(
    row_mask: np.ndarray,
    candidates: list[tuple[int, int, int, int]],
) -> int | None:
    """
    自动分割失败时按整行宽高比推断位数。
    三位数被内部泛光粘成一块时，投影法常只得到 1 个候选框。
    """
    if len(candidates) >= 2:
        return None

    ys, xs = np.where(row_mask > 0)
    if xs.size == 0 or ys.size == 0:
        return None

    width = int(xs.max() - xs.min() + 1)
    height = int(ys.max() - ys.min() + 1)
    if height <= 0:
        return None

    aspect = width / height
    if aspect >= 2.1:
        return 3
    if aspect >= 1.35:
        return 2
    return None


def split_digits_by_width(row_mask: np.ndarray, expected_digits: int) -> list[tuple[int, int, int, int]]:
    """
    等宽均分法切分数字（自动投影分割失败时的兜底方案）。
    将整行 mask 按宽度均分为 expected_digits 份，每份内收紧到实际笔画边界。
    """
    ys, xs = np.where(row_mask > 0)
    if xs.size == 0 or ys.size == 0:
        return []

    # 整行总包围框
    x1, x2 = int(xs.min()), int(xs.max() + 1)
    y1, y2 = int(ys.min()), int(ys.max() + 1)
    total_w = x2 - x1
    if total_w <= 0:
        return []

    boxes = []
    for index in range(expected_digits):
        # 均分宽度
        bx1 = x1 + round(total_w * index / expected_digits)
        bx2 = x1 + round(total_w * (index + 1) / expected_digits)
        # 在该份内收紧到实际笔画范围
        digit_mask = row_mask[y1:y2, bx1:bx2]
        dys, dxs = np.where(digit_mask > 0)
        if dxs.size == 0 or dys.size == 0:
            boxes.append((bx1, y1, bx2, y2))
            continue
        boxes.append((bx1 + int(dxs.min()), y1 + int(dys.min()),
                      bx1 + int(dxs.max() + 1), y1 + int(dys.max() + 1)))
    return boxes


# ============================================================
# 单数字识别：七段采样 + 模板匹配
# ============================================================

def recognize_digit(mask: np.ndarray, box: tuple[int, int, int, int]) -> tuple[str, dict[str, float]]:
    """
    识别 mask 中指定包围框内的单个数字。

    算法：
      1. 从 mask 中裁剪出数字区域。
      2. 添加 padding 防止边缘段被切除。
      3. 缩放到固定 40×70 尺寸，让段采样坐标与数字位置对齐。
      4. 对归一化后每个段区域计算白色像素占比 → 该段"点亮"的置信度。
      5. 以最亮段的置信度为基准做归一化（避免整体偏暗影响匹配）。
      6. 对 0~9 每个数字计算加权欧氏距离：
         - 目标=1的段（该亮点亮）其偏差权重更高(1.0)
         - 目标=0的段（该亮熄灭）其偏差权重略低(0.75)
         - 这是因为"熄灭"的判断更不可靠（可能有相邻段漏光）
      7. 取距离最小的数字为识别结果，同时计算 margin（与第二名的差距）
         用于评估识别结果的可信度。
    """
    x1, y1, x2, y2 = box
    digit = mask[y1:y2, x1:x2]
    if digit.size == 0:
        return "?", {}

    # 添加 padding，避免边缘段丢失
    pad_x = max(1, round((x2 - x1) * 0.10))
    pad_y = max(1, round((y2 - y1) * 0.08))
    digit = cv2.copyMakeBorder(digit, pad_y, pad_y, pad_x, pad_x, cv2.BORDER_CONSTANT, value=0)

    # 缩放到固定尺寸（NEAREST 保证二值 mask 不被插值模糊）
    digit = cv2.resize(digit, (40, 70), interpolation=cv2.INTER_NEAREST)
    # 轻微膨胀，补上缩放后变细的笔画
    digit = cv2.dilate(digit, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)

    # ---- 七段 ROI 区域定义 ----
    # 在 40×70 的画布上，人工标定 a~g 每段的大致矩形区域
    # 坐标格式: (x1, y1, x2, y2)
    #
    #          aaaaaaaa
    #         (9,2)-(31,13)
    #       f  |        |  b
    #    (1,9) |  gggg  | (27,9)
    #    -(13,34)  (9,29)-(31,42) (39,34)-
    #       e  |        |  c
    #    (1,36) |        | (27,36)
    #    -(13,61)  dddd  -(39,61)
    #         (9,57)-(31,69)
    #
    segment_boxes = {
        "a": (11, 3, 29, 10),   # 顶部横段
        "b": (31, 11, 37, 31),  # 右上竖段
        "c": (31, 39, 37, 59),  # 右下竖段
        "d": (11, 60, 29, 67),  # 底部横段
        "e": (3, 39, 9, 59),    # 左下竖段
        "f": (3, 11, 9, 31),    # 左上竖段
        "g": (11, 31, 29, 39),  # 中间横段
    }

    # 计算每个段区域的白色像素占比（= 该段被点亮的概率）
    scores: dict[str, float] = {}
    for name, (sx1, sy1, sx2, sy2) in segment_boxes.items():
        roi = digit[sy1:sy2, sx1:sx2]
        scores[name] = float(np.count_nonzero(roi) / max(1, roi.size))

    # 获取最亮段的分数作为归一化基准
    max_score = max(scores.values()) if scores else 0.0
    if max_score <= 0.0:
        return "?", scores

    # 归一化到 [0, 1]，让不同亮度的数字能统一比较
    normalized_scores = {
        segment: min(1.0, score / max_score)
        for segment, score in scores.items()
    }

    # ---- 与标准段码匹配 ----
    best_digit = "?"
    best_distance = float("inf")
    second_distance = float("inf")
    for number, enabled_segments in SEGMENT_MAP.items():
        enabled = set(enabled_segments)
        distance = 0.0
        for seg in SEGMENT_ORDER:
            score = normalized_scores[seg]
            target = 1.0 if seg in enabled else 0.0
            # 预期亮的段权重 1.0，预期灭的段权重 0.75
            # 降低"预期灭"段权重的理由：相邻段可能漏光造成误检
            weight = 1.0 if target else 0.75
            distance += weight * (score - target) ** 2
        if distance < best_distance:
            second_distance = best_distance
            best_distance = distance
            best_digit = number
        elif distance < second_distance:
            second_distance = distance

    # 附加诊断信息（debug 模式使用）
    scores["_distance"] = best_distance   # 最佳匹配的距离
    scores["_margin"] = second_distance - best_distance  # 与第二名的差距（越大越可信）
    scores["_max_score"] = max_score
    return best_digit, scores


# ============================================================
# 完整识别流程
# ============================================================

def recognize_image(image: np.ndarray, cfg: DetectConfig) -> tuple[list[RowResult], np.ndarray, np.ndarray]:
    """
    对一张图片执行完整的七段数码管识别流水线。

    流程：
      scale 放大 → build_led_mask 提取笔画 → find_rows 定位行
      → split_digits 切分每位数字 → recognize_digit 逐位识别

    返回：(识别结果列表, 缩放后的原图, 二值mask)
    """
    # 放大图像，让细节更清晰（小ROI必须放大）
    if cfg.scale != 1.0:
        image = cv2.resize(image, None, fx=cfg.scale, fy=cfg.scale, interpolation=cv2.INTER_CUBIC)

    results: list[RowResult] = []

    masks = [build_led_mask(image, cfg)]       # 提取亮笔画
    if cfg.color_mode == "led":
        # 默认场景是上红/粉、下黄绿。分颜色找行可以避免上下两行因泛光粘连成一行。
        masks = [
            build_led_mask(image, cfg, color_mode="red"),
            build_led_mask(image, cfg, color_mode="green"),
        ]

    preview_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    for mask in masks:
        preview_mask = cv2.bitwise_or(preview_mask, mask)
        rows = find_rows(mask, cfg)             # 定位行

        if len(rows) > 1:
            rows = sorted(
                rows,
                key=lambda box: np.count_nonzero(mask[box[1]:box[3], box[0]:box[2]]),
                reverse=True,
            )[:1]

        for row_box in rows:
            x1, y1, x2, y2 = row_box
            row_mask = mask[y1:y2, x1:x2]                       # 截取该行的 mask
            digit_boxes = split_digits(row_mask, (x1, y1), cfg)  # 切分每位数字
            number_chars = []
            digit_scores = []
            for digit_box in digit_boxes:
                number, scores = recognize_digit(mask, digit_box)  # 识别每位数字
                number_chars.append(number)
                digit_scores.append(scores)
            results.append(RowResult("".join(number_chars), row_box, digit_boxes, digit_scores))

    results = sorted(results, key=lambda result: result.row_box[1])
    return results, image, preview_mask


# ============================================================
# 可视化与结果输出
# ============================================================

def draw_preview(image: np.ndarray, results: list[RowResult]) -> np.ndarray:
    """
    在原图上绘制识别结果：
      - 黄色框 + 标签：行级框
      - 青色框 + 数字：每位数字的框与识别结果
    """
    preview = image.copy()
    for row_index, result in enumerate(results, start=1):
        x1, y1, x2, y2 = result.row_box
        # 行级包围框（黄色）
        cv2.rectangle(preview, (x1, y1), (x2, y2), (255, 180, 0), 1)
        cv2.putText(
            preview,
            f"row{row_index}: {result.number}",
            (x1, max(14, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 0),
            1,
            cv2.LINE_AA,
        )

        # 每位数字的包围框（青色）
        for digit_box, digit in zip(result.digit_boxes, result.number):
            dx1, dy1, dx2, dy2 = digit_box
            cv2.rectangle(preview, (dx1, dy1), (dx2, dy2), (0, 255, 255), 1)
            cv2.putText(
                preview,
                digit,
                (dx1, dy2 + 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )
    return preview


def print_result(path: str, results: list[RowResult], cfg: DetectConfig) -> None:
    """在控制台打印识别结果，debug 模式下额外输出每个数字的段分数和匹配距离。"""
    print(f"\n[IMAGE] {path}")
    if not results:
        print("[WARN] 未检测到数字行，可以尝试降低 --min-v 或 --row-threshold-ratio")
        return

    for row_index, result in enumerate(results, start=1):
        print(f"  第{row_index}行: {result.number or '(未识别)'}")
        if cfg.debug:
            print(f"    行框: {result.row_box}")
            for digit_index, (box, scores) in enumerate(
                zip(result.digit_boxes, result.digit_scores),
                start=1,
            ):
                # 只输出 a~g 段的分数
                core_scores = ", ".join(f"{k}:{scores[k]:.2f}" for k in SEGMENT_ORDER)
                print(
                    f"    第{digit_index}位框: {box}, {core_scores}, "
                    f"distance:{scores.get('_distance', 0):.3f}, "
                    f"margin:{scores.get('_margin', 0):.3f}"
                )


# ============================================================
# 命令行参数与主入口
# ============================================================

def parse_args() -> argparse.Namespace:
    """解析命令行参数。所有阈值参数都有合理的默认值。"""
    parser = argparse.ArgumentParser(description="七段数码管 ROI 图片识别 Demo")
    parser.add_argument("--image", "-i", nargs="*", default=None,
                        help="ROI 图片路径；不传则弹出文件选择框，可一次选择多张")
    parser.add_argument("--scale", type=float, default=4.0,
                        help="识别前放大倍数，默认 4.0，适合很小的 ROI")
    parser.add_argument("--min-v", type=int, default=85,
                        help="HSV 亮度阈值，越低越容易保留暗笔画")
    parser.add_argument("--min-s", type=int, default=30,
                        help="HSV 饱和度阈值，用于彩色数码管")
    parser.add_argument("--min-chroma", type=int, default=40,
                        help="BGR 色度差阈值，用于排除灰底高光")
    parser.add_argument("--color-percentile", type=float, default=75.0,
                        help="候选颜色内的强度分位数阈值，越高越能压掉内部泛光")
    parser.add_argument("--white-v", type=int, default=180,
                        help="低饱和白色/浅色笔画的亮度阈值，仅 --include-white 时启用")
    parser.add_argument("--color-mode", choices=("led", "red", "green", "any"), default="led",
                        help="笔画颜色模式：led=红/粉+黄绿，red=只识别上方红/粉，green=只识别下方黄绿，any=任意高饱和彩色")
    parser.add_argument("--include-white", action="store_true",
                        help="额外保留低饱和高亮白色笔画；彩色数码管默认不要开启")
    parser.add_argument("--row-threshold-ratio", type=float, default=0.12,
                        help="行投影阈值比例")
    parser.add_argument("--col-threshold-ratio", type=float, default=0.10,
                        help="列投影阈值比例")
    parser.add_argument("--expected-digits", type=int, choices=(2, 3), default=None,
                        help="指定每行数字位数；不指定则自动按列分割")
    parser.add_argument("--show", action="store_true",
                        help="显示识别框和二值化结果")
    parser.add_argument("--debug", action="store_true",
                        help="输出每个数字的七段占用率和匹配距离")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    # 优先使用命令行传入的图片路径，否则弹文件选择框
    paths = args.image if args.image else choose_images()
    if not paths:
        print("[INFO] 未选择图片")
        return 0

    cfg = DetectConfig(
        scale=args.scale,
        min_v=args.min_v,
        min_s=args.min_s,
        min_chroma=args.min_chroma,
        color_percentile=args.color_percentile,
        white_v=args.white_v,
        color_mode=args.color_mode,
        include_white=args.include_white,
        row_threshold_ratio=args.row_threshold_ratio,
        col_threshold_ratio=args.col_threshold_ratio,
        expected_digits=args.expected_digits,
        show=args.show,
        debug=args.debug,
    )

    for path in paths:
        normalized_path = os.fspath(Path(path))
        image = read_image(normalized_path)
        if image is None:
            print(f"[ERROR] 图片读取失败: {normalized_path}")
            continue

        # 执行识别流程
        results, scaled_image, mask = recognize_image(image, cfg)
        print_result(normalized_path, results, cfg)

        if cfg.show:
            preview = draw_preview(scaled_image, results)
            cv2.imshow("seven-segment preview", preview)
            cv2.imshow("seven-segment mask", mask)
            print("[INFO] 按任意键继续下一张，按 q 退出")
            key = cv2.waitKey(0) & 0xFF
            if key == ord("q"):
                break

    if cfg.show:
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
