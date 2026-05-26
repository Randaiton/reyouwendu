from __future__ import annotations

import argparse
import os
import random
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Any

from rec_utils import (
    DEFAULT_CONFIG_PATH,
    apply_preprocess,
    crop_hotzone,
    get_hotzone,
    get_hotzone_order,
    get_path,
    list_image_files,
    make_dataset_image_name,
    parse_labels_from_filename,
    path_for_config,
    read_image,
    read_yaml,
    require_box,
    write_image,
    write_yaml,
)


def prepare_assets(config: dict[str, Any], force: bool) -> None:
    """
    显式准备 PaddleOCR 源码和 PP-OCRv5 mobile rec 预训练权重。

    参数:
        config: YAML 配置对象。
        force: 是否允许覆盖已有资源。
    """
    paddleocr_dir = get_path(config, "paddleocr_source")
    pretrained_dir = get_path(config, "pretrained_dir")
    pretrained_dir.mkdir(parents=True, exist_ok=True)

    if paddleocr_dir.exists() and force:
        shutil.rmtree(paddleocr_dir)

    if not (paddleocr_dir / "tools" / "train.py").exists():
        paddleocr_dir.parent.mkdir(parents=True, exist_ok=True)
        clone_cmd = [
            "git",
            "clone",
            "--depth",
            "1",
            "https://github.com/PaddlePaddle/PaddleOCR.git",
            str(paddleocr_dir),
        ]
        print("[RUN]", " ".join(clone_cmd))
        subprocess.run(clone_cmd, check=True)
    else:
        print(f"[SKIP] PaddleOCR 源码已存在: {paddleocr_dir}")

    pretrained_path = pretrained_dir / config["model"]["pretrained_file"]
    if pretrained_path.exists() and not force:
        print(f"[SKIP] 预训练权重已存在: {pretrained_path}")
        return

    print(f"[DOWNLOAD] {config['model']['pretrained_url']}")
    urllib.request.urlretrieve(config["model"]["pretrained_url"], pretrained_path)
    print(f"[DONE] 预训练权重已保存: {pretrained_path}")


def clean_dataset(dataset_dir: Path) -> None:
    """
    清理数据集目录。

    参数:
        dataset_dir: rec 数据集目录。
    """
    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)


def build_dataset(config: dict[str, Any], clean: bool) -> tuple[int, int]:
    """
    根据 8 个热区裁剪原始图片，并生成 PaddleOCR rec 数据清单。

    参数:
        config: YAML 配置对象。
        clean: 是否先清理旧数据集目录。

    返回:
        tuple[int, int]: (有效样本数, 跳过图片数)。
    """
    raw_dir = get_path(config, "raw_frames")
    dataset_dir = get_path(config, "rec_dataset")
    images_dir = dataset_dir / "images"
    if clean:
        clean_dataset(dataset_dir)
    images_dir.mkdir(parents=True, exist_ok=True)

    image_paths = list_image_files(raw_dir, config["dataset"]["image_extensions"])
    if not image_paths:
        raise RuntimeError(f"未找到原始训练图片: {raw_dir}")

    order = get_hotzone_order(config)
    entries: list[tuple[str, str]] = []
    skipped = 0

    for source_path in image_paths:
        try:
            labels = parse_labels_from_filename(source_path, expected_count=len(order))
        except ValueError as exc:
            skipped += 1
            print(f"[SKIP] {exc}")
            continue

        image = read_image(source_path)
        if image is None:
            skipped += 1
            print(f"[SKIP] 图片读取失败: {source_path}")
            continue

        for hotzone_name, label in zip(order, labels):
            hotzone = get_hotzone(config, hotzone_name)
            box = require_box(hotzone, hotzone_name)
            crop = crop_hotzone(image, box)
            crop = apply_preprocess(crop, hotzone.get("preprocess"))
            crop_name = make_dataset_image_name(source_path, hotzone_name)
            crop_path = images_dir / crop_name
            if not write_image(crop_path, crop):
                raise RuntimeError(f"裁剪图保存失败: {crop_path}")

            relative_path = crop_path.relative_to(dataset_dir).as_posix()
            entries.append((relative_path, label))

    if not entries:
        raise RuntimeError("未生成任何有效 rec 样本，请检查文件名标签和热区坐标")

    write_dataset_lists(config, dataset_dir, entries)
    print(f"[DONE] 生成 rec 样本 {len(entries)} 条，跳过原图 {skipped} 张")
    print(f"[DONE] 数据集目录: {dataset_dir}")
    return len(entries), skipped


def write_dataset_lists(config: dict[str, Any], dataset_dir: Path, entries: list[tuple[str, str]]) -> None:
    """
    写入 train_list.txt 和 val_list.txt。

    参数:
        config: YAML 配置对象。
        dataset_dir: rec 数据集目录。
        entries: (相对图片路径, 标签) 列表。
    """
    random_seed = int(config["dataset"].get("random_seed", 20260526))
    val_ratio = float(config["dataset"].get("val_ratio", 0.2))
    items = entries[:]
    random.Random(random_seed).shuffle(items)

    if len(items) <= 1:
        val_count = 0
    else:
        val_count = max(1, round(len(items) * val_ratio))
        val_count = min(val_count, len(items) - 1)

    val_items = items[:val_count]
    train_items = items[val_count:]

    for list_path, rows in (
        (dataset_dir / "train_list.txt", train_items),
        (dataset_dir / "val_list.txt", val_items or train_items[:1]),
    ):
        list_path.parent.mkdir(parents=True, exist_ok=True)
        with list_path.open("w", encoding="utf-8", newline="\n") as f:
            for image_path, label in rows:
                f.write(f"{image_path}\t{label}\n")

    print(f"[DONE] train_list: {len(train_items)} 条")
    print(f"[DONE] val_list: {len(val_items or train_items[:1])} 条")


def load_official_train_config(config: dict[str, Any]) -> dict[str, Any]:
    """
    读取官方 PaddleOCR rec 训练配置。

    参数:
        config: YAML 配置对象。

    返回:
        dict: 官方配置内容。
    """
    source_dir = get_path(config, "paddleocr_source")
    official_config = source_dir / config["model"]["official_config"]
    if not official_config.exists():
        raise RuntimeError(
            "未找到官方 PP-OCRv5_mobile_rec.yml，请先显式运行 --prepare-assets，"
            f"或确认路径存在: {official_config}"
        )
    return read_yaml(official_config)


def make_simple_rec_transforms(with_aug: bool, max_text_length: int) -> list[dict[str, Any]]:
    """
    生成适合小型数字 rec 数据集的 transforms。

    参数:
        with_aug: 是否启用 RecAug。
        max_text_length: 最大文本长度。

    返回:
        list[dict]: PaddleOCR transforms 配置。
    """
    transforms: list[dict[str, Any]] = [
        {"DecodeImage": {"img_mode": "BGR", "channel_first": False}},
    ]
    if with_aug:
        transforms.append({"RecAug": None})
    transforms.extend([
        {"MultiLabelEncode": {"gtc_encode": "NRTRLabelEncode"}},
        {"RecResizeImg": {"image_shape": [3, 48, 320]}},
        {
            "KeepKeys": {
                "keep_keys": [
                    "image",
                    "label_ctc",
                    "label_gtc",
                    "length",
                    "valid_ratio",
                ]
            }
        },
    ])
    return transforms


def update_architecture_max_text_length(train_config: dict[str, Any], max_text_length: int) -> None:
    """
    同步 PP-OCRv5 识别头内部的最大文本长度。

    参数:
        train_config: PaddleOCR 训练配置。
        max_text_length: 最大文本长度，本项目固定为 2-3 位数字场景。
    """
    head = train_config.get("Architecture", {}).get("Head", {})
    for item in head.get("head_list", []):
        nrtr_head = item.get("NRTRHead") if isinstance(item, dict) else None
        if isinstance(nrtr_head, dict):
            nrtr_head["max_text_length"] = max_text_length


def generate_training_config(config: dict[str, Any], run_name: str, epoch_num: int | None) -> Path:
    """
    从官方配置派生 CPU 微调配置。

    参数:
        config: YAML 配置对象。
        run_name: 本次训练名称。
        epoch_num: 可选 epoch 覆盖值。

    返回:
        Path: 生成的训练配置路径。
    """
    train_config = load_official_train_config(config)
    dataset_dir = get_path(config, "rec_dataset")
    runs_dir = get_path(config, "runs") / run_name
    model_dir = get_path(config, "models") / config["model"]["inference_dir_name"]
    pretrained_path = get_path(config, "pretrained_dir") / config["model"]["pretrained_file"]
    max_text_length = int(config["training"].get("max_text_length", 3))

    global_cfg = train_config.setdefault("Global", {})
    global_cfg["model_name"] = config["model"]["name"]
    global_cfg["use_gpu"] = False
    global_cfg["distributed"] = False
    global_cfg["epoch_num"] = int(epoch_num or config["training"].get("epoch_num", 20))
    global_cfg["print_batch_step"] = int(config["training"].get("print_batch_step", 10))
    global_cfg["save_epoch_step"] = int(config["training"].get("save_epoch_step", 5))
    global_cfg["eval_batch_step"] = config["training"].get("eval_batch_step", [0, 100])
    global_cfg["pretrained_model"] = path_for_config(pretrained_path)
    global_cfg["save_model_dir"] = path_for_config(runs_dir / "checkpoints")
    global_cfg["save_inference_dir"] = path_for_config(model_dir)
    global_cfg["save_res_path"] = path_for_config(runs_dir / "predicts.txt")
    global_cfg["max_text_length"] = max_text_length
    global_cfg["use_space_char"] = False
    update_architecture_max_text_length(train_config, max_text_length)

    train_config.setdefault("Optimizer", {}).setdefault("lr", {})["learning_rate"] = float(
        config["training"].get("learning_rate", 0.0001)
    )

    train_cfg = train_config.setdefault("Train", {})
    train_cfg.pop("sampler", None)
    train_cfg["dataset"] = {
        "name": "SimpleDataSet",
        "data_dir": path_for_config(dataset_dir),
        "label_file_list": [path_for_config(dataset_dir / "train_list.txt")],
        "transforms": make_simple_rec_transforms(with_aug=True, max_text_length=max_text_length),
    }
    train_cfg["loader"] = {
        "shuffle": True,
        "drop_last": False,
        "batch_size_per_card": int(config["training"].get("batch_size_per_card", 16)),
        "num_workers": int(config["training"].get("num_workers", 0)),
    }

    eval_cfg = train_config.setdefault("Eval", {})
    eval_cfg["dataset"] = {
        "name": "SimpleDataSet",
        "data_dir": path_for_config(dataset_dir),
        "label_file_list": [path_for_config(dataset_dir / "val_list.txt")],
        "transforms": make_simple_rec_transforms(with_aug=False, max_text_length=max_text_length),
    }
    eval_cfg["loader"] = {
        "shuffle": False,
        "drop_last": False,
        "batch_size_per_card": int(config["training"].get("batch_size_per_card", 16)),
        "num_workers": int(config["training"].get("num_workers", 0)),
    }

    runs_dir.mkdir(parents=True, exist_ok=True)
    generated_config_path = runs_dir / "config.yml"
    write_yaml(generated_config_path, train_config)
    print(f"[DONE] 训练配置已生成: {generated_config_path}")
    return generated_config_path


def default_weights_path(config: dict[str, Any], run_name: str) -> Path:
    """
    获取默认训练权重前缀。

    参数:
        config: YAML 配置对象。
        run_name: 本次训练名称。

    返回:
        Path: 默认 best_accuracy 权重前缀。
    """
    return get_path(config, "runs") / run_name / "checkpoints" / "best_accuracy"


def run_paddle_tool(config: dict[str, Any], tool_name: str, args: list[str]) -> None:
    """
    调用 PaddleOCR 官方 tools 下的脚本。

    参数:
        config: YAML 配置对象。
        tool_name: tools 下的脚本名称。
        args: 额外命令参数。
    """
    source_dir = get_path(config, "paddleocr_source")
    tool_path = source_dir / "tools" / tool_name
    if not tool_path.exists():
        raise RuntimeError(f"PaddleOCR 工具不存在: {tool_path}")

    command = [sys.executable, str(tool_path), *args]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    print("[RUN]", " ".join(command))
    subprocess.run(command, cwd=source_dir, env=env, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PaddleOCR-v5 mobile rec CPU 微调入口")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH),
                        help="YAML 配置文件路径")
    parser.add_argument("--run-name",
                        help="训练运行名称；默认读取配置 project.run_name")
    parser.add_argument("--prepare-assets", action="store_true",
                        help="显式下载/准备 PaddleOCR 源码和预训练权重")
    parser.add_argument("--build-dataset", action="store_true",
                        help="根据 8 个热区裁剪原始图片并生成 rec 数据集")
    parser.add_argument("--train", action="store_true",
                        help="显式启动 CPU 微调训练")
    parser.add_argument("--eval", action="store_true",
                        help="显式启动评估")
    parser.add_argument("--export", action="store_true",
                        help="显式导出静态推理模型")
    parser.add_argument("--clean-dataset", action="store_true",
                        help="构建数据集前清理旧 data/rec_dataset")
    parser.add_argument("--force-assets", action="store_true",
                        help="准备资源时覆盖已有 PaddleOCR 源码和预训练权重")
    parser.add_argument("--epoch-num", type=int,
                        help="临时覆盖训练 epoch 数，适合 smoke test")
    parser.add_argument("--weights",
                        help="评估/导出使用的权重前缀或 .pdparams 路径；默认使用 best_accuracy")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = read_yaml(args.config)
    run_name = args.run_name or config.get("project", {}).get("run_name", "ppocrv5_mobile_rec_cpu")

    if not any([args.prepare_assets, args.build_dataset, args.train, args.eval, args.export]):
        print("[INFO] 未指定动作。可使用 --build-dataset、--train、--eval、--export 或 --prepare-assets")
        return 0

    try:
        if args.prepare_assets:
            prepare_assets(config, force=args.force_assets)

        if args.build_dataset:
            build_dataset(config, clean=args.clean_dataset)

        generated_config_path = None
        if args.train or args.eval or args.export:
            generated_config_path = generate_training_config(config, run_name, args.epoch_num)

        if args.train:
            run_paddle_tool(config, "train.py", ["-c", path_for_config(generated_config_path)])

        weights = Path(args.weights) if args.weights else default_weights_path(config, run_name)
        if args.eval:
            run_paddle_tool(
                config,
                "eval.py",
                [
                    "-c",
                    path_for_config(generated_config_path),
                    "-o",
                    f"Global.pretrained_model={path_for_config(weights)}",
                ],
            )

        if args.export:
            model_dir = get_path(config, "models") / config["model"]["inference_dir_name"]
            run_paddle_tool(
                config,
                "export_model.py",
                [
                    "-c",
                    path_for_config(generated_config_path),
                    "-o",
                    f"Global.pretrained_model={path_for_config(weights)}",
                    f"Global.save_inference_dir={path_for_config(model_dir)}",
                ],
            )
    except Exception as exc:
        print(f"[ERROR] {exc}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
