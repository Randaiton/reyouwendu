"""
整行数字数据标注工具。

用法：
  python label_rows.py --input screenshots

操作：
  1. OpenCV 窗口中框选一整行数字，按 Enter/Space 确认。
  2. 在控制台输入该行标签，例如 223、300。
  3. 标签输入 u 撤销上一条，s 跳过当前行，q 保存并退出。
"""
from __future__ import annotations

import argparse
import csv
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2

from cnn_ocr_common import list_image_files, normalize_label, read_image, relative_path, write_image


LABEL_COLUMNS = ["image_path", "label", "source_roi", "row_index", "x", "y", "w", "h"]


@dataclass
class LabelState:
    labels_path: Path
    rows_dir: Path
    raw_dir: Path
    project_dir: Path
    rows: list[dict[str, str]]
    next_index: int


def load_rows(labels_path: Path) -> list[dict[str, str]]:
    if not labels_path.exists():
        return []
    with labels_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return [row for row in reader if row]


def save_rows(labels_path: Path, rows: list[dict[str, str]]) -> None:
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    with labels_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LABEL_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def find_next_index(rows_dir: Path) -> int:
    max_index = 0
    for path in rows_dir.glob("row_*.png"):
        try:
            max_index = max(max_index, int(path.stem.split("_")[-1]))
        except ValueError:
            continue
    return max_index + 1


def copy_source_image(source_path: Path, state: LabelState) -> Path:
    state.raw_dir.mkdir(parents=True, exist_ok=True)
    target_name = f"{source_path.parent.name}_{source_path.name}"
    target_path = state.raw_dir / target_name
    if not target_path.exists():
        shutil.copy2(source_path, target_path)
    return target_path


def undo_last(state: LabelState) -> None:
    if not state.rows:
        print("[INFO] 没有可撤销的标注")
        return

    row = state.rows.pop()
    image_path = state.project_dir / row["image_path"]
    if image_path.exists():
        image_path.unlink()
    save_rows(state.labels_path, state.rows)
    print(f"[UNDO] 已撤销: {row['image_path']} -> {row['label']}")


def append_label(
    state: LabelState,
    source_path: Path,
    source_image_path: Path,
    image,
    row_index: int,
    roi: tuple[int, int, int, int],
    label: str,
) -> None:
    x, y, w, h = roi
    crop = image[y:y + h, x:x + w]
    row_path = state.rows_dir / f"row_{state.next_index:06d}.png"
    state.next_index += 1

    if not write_image(row_path, crop):
        raise RuntimeError(f"行图片保存失败: {row_path}")

    row = {
        "image_path": relative_path(row_path, state.project_dir),
        "label": label,
        "source_roi": relative_path(source_image_path, state.project_dir),
        "row_index": str(row_index),
        "x": str(x),
        "y": str(y),
        "w": str(w),
        "h": str(h),
    }
    state.rows.append(row)
    save_rows(state.labels_path, state.rows)
    print(f"[SAVE] {row['image_path']} -> {label}，来源: {source_path.name} 第{row_index + 1}行")


def label_image(source_path: Path, state: LabelState, rows_per_image: int) -> bool:
    image = read_image(source_path)
    if image is None:
        print(f"[WARN] 图片读取失败，跳过: {source_path}")
        return True

    source_image_path = copy_source_image(source_path, state)
    window_name = f"label_rows - {source_path.name}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    for row_index in range(rows_per_image):
        print(f"\n[IMAGE] {source_path}")
        print(f"[INFO] 请框选第 {row_index + 1}/{rows_per_image} 行数字，Enter/Space 确认，Esc 取消本行")
        roi = cv2.selectROI(window_name, image, showCrosshair=True, fromCenter=False)
        x, y, w, h = [int(value) for value in roi]
        if w <= 0 or h <= 0:
            print("[SKIP] 当前行未框选")
            continue

        while True:
            value = input("输入标签(2-3位数字)，u=撤销上一条，s=跳过本行，q=保存退出: ").strip()
            if value.lower() == "q":
                cv2.destroyWindow(window_name)
                return False
            if value.lower() == "u":
                undo_last(state)
                continue
            if value.lower() == "s" or value == "":
                print("[SKIP] 当前行已跳过")
                break

            try:
                label = normalize_label(value)
            except ValueError as exc:
                print(f"[ERROR] {exc}")
                continue

            append_label(state, source_path, source_image_path, image, row_index, (x, y, w, h), label)
            break

    cv2.destroyWindow(window_name)
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="整行数字 CNN 数据标注工具")
    parser.add_argument("--input", "-i", required=True,
                        help="待标注图片或目录，目录会递归读取 jpg/png/bmp")
    parser.add_argument("--dataset-dir", default="dataset",
                        help="数据集目录，默认 dataset")
    parser.add_argument("--rows-per-image", type=int, default=2,
                        help="每张 ROI 默认标注的行数，默认 2")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_dir = Path.cwd()
    dataset_dir = Path(args.dataset_dir)
    rows_dir = dataset_dir / "rows"
    raw_dir = dataset_dir / "raw"
    labels_path = dataset_dir / "labels.csv"

    state = LabelState(
        labels_path=labels_path,
        rows_dir=rows_dir,
        raw_dir=raw_dir,
        project_dir=project_dir,
        rows=load_rows(labels_path),
        next_index=find_next_index(rows_dir),
    )

    files = list_image_files(args.input)
    if not files:
        print(f"[ERROR] 未找到图片: {args.input}")
        return 1

    print("[INFO] 标注会实时保存到 dataset/labels.csv")
    print("[INFO] 标签只接受 2-3 位数字；输入 q 可保存并退出")

    for source_path in files:
        keep_going = label_image(source_path, state, args.rows_per_image)
        if not keep_going:
            break

    save_rows(state.labels_path, state.rows)
    cv2.destroyAllWindows()
    print(f"\n[DONE] 已保存 {len(state.rows)} 条标注: {state.labels_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
