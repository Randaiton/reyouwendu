"""
训练整行数字 CRNN，并导出 ONNX。

用法：
  python train_row_crnn.py --labels dataset/labels.csv --out models/row_crnn.onnx

说明：
  训练依赖在 requirements-train.txt 中，由用户手动安装。
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

import cv2
import numpy as np

from cnn_ocr_common import (
    BLANK_INDEX,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    NUM_CLASSES,
    decode_ctc_logits,
    encode_label,
    preprocess_row_image,
    read_image,
)


LABEL_COLUMNS = ["image_path", "label", "source_roi", "row_index", "x", "y", "w", "h"]


try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
except ImportError as exc:
    print("[ERROR] 缺少训练依赖，请先手动安装 requirements-train.txt")
    print(f"[ERROR] {exc}")
    sys.exit(1)


class RowCRNN(nn.Module):
    """轻量 CNN-CTC：CNN 提取特征，按宽度方向输出 CTC 序列。"""

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (2, 1)),

            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (2, 1)),

            nn.Conv2d(128, 160, 3, padding=1),
            nn.BatchNorm2d(160),
            nn.ReLU(inplace=True),
        )
        self.temporal = nn.Sequential(
            nn.Conv1d(160, 192, 3, padding=1),
            nn.BatchNorm1d(192),
            nn.ReLU(inplace=True),
            nn.Conv1d(192, 192, 3, padding=1),
            nn.BatchNorm1d(192),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Conv1d(192, num_classes, 1)

    def forward(self, x):
        features = self.features(x)
        features = features.mean(dim=2)
        sequence = self.temporal(features)
        logits = self.classifier(sequence)
        return logits.permute(0, 2, 1)


class RowDataset(Dataset):
    def __init__(self, rows: list[dict[str, str]], project_dir: Path, augment: bool):
        self.rows = rows
        self.project_dir = project_dir
        self.augment = augment

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        image_path = resolve_path(row["image_path"], self.project_dir)
        image = read_image(image_path)
        if image is None:
            raise RuntimeError(f"图片读取失败: {image_path}")

        if self.augment:
            image = augment_image(image)

        tensor = torch.from_numpy(preprocess_row_image(image)).float()
        target = torch.tensor(encode_label(row["label"]), dtype=torch.long)
        return tensor, target, row["label"]


def resolve_path(path_text: str, project_dir: Path) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return project_dir / path


def augment_image(image: np.ndarray) -> np.ndarray:
    output = image.copy()

    alpha = random.uniform(0.75, 1.25)
    beta = random.uniform(-25, 25)
    output = cv2.convertScaleAbs(output, alpha=alpha, beta=beta)

    if random.random() < 0.25:
        output = cv2.GaussianBlur(output, (3, 3), 0)

    if random.random() < 0.25:
        noise = np.random.normal(0, 6, output.shape).astype(np.int16)
        output = np.clip(output.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return output


def load_label_rows(labels_path: Path) -> list[dict[str, str]]:
    with labels_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            encode_label(row["label"])
            rows.append(row)
    return rows


def write_split(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = list(rows[0].keys()) if rows else LABEL_COLUMNS
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_split(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def load_or_create_splits(
    labels_path: Path,
    all_rows: list[dict[str, str]],
    seed: int,
) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    split_dir = labels_path.parent / "splits"
    train_path = split_dir / "train.csv"
    val_path = split_dir / "val.csv"
    test_path = split_dir / "test.csv"

    if train_path.exists() and val_path.exists() and test_path.exists():
        return read_split(train_path), read_split(val_path), read_split(test_path)

    rows = all_rows[:]
    random.Random(seed).shuffle(rows)
    total = len(rows)
    train_end = max(1, int(total * 0.70))
    val_end = max(train_end + 1, int(total * 0.85)) if total >= 3 else train_end

    train_rows = rows[:train_end]
    val_rows = rows[train_end:val_end]
    test_rows = rows[val_end:]

    if not val_rows and len(train_rows) > 1:
        val_rows = [train_rows.pop()]
    if not test_rows and len(train_rows) > 1:
        test_rows = [train_rows.pop()]

    write_split(train_path, train_rows)
    write_split(val_path, val_rows)
    write_split(test_path, test_rows)
    return train_rows, val_rows, test_rows


def collate_batch(batch):
    images, targets, labels = zip(*batch)
    images = torch.stack(images, dim=0)
    target_lengths = torch.tensor([len(target) for target in targets], dtype=torch.long)
    targets = torch.cat(targets, dim=0)
    return images, targets, target_lengths, labels


def evaluate(model, loader, device) -> tuple[float, float]:
    model.eval()
    total_rows = 0
    correct_rows = 0
    total_chars = 0
    correct_chars = 0

    with torch.no_grad():
        for images, _, _, labels in loader:
            images = images.to(device)
            logits = model(images).cpu().numpy()
            predictions = decode_ctc_logits(logits)

            for prediction, label in zip(predictions, labels):
                total_rows += 1
                correct_rows += int(prediction == label)
                total_chars += max(len(label), len(prediction))
                correct_chars += sum(1 for a, b in zip(prediction, label) if a == b)

    row_acc = correct_rows / total_rows if total_rows else 0.0
    char_acc = correct_chars / total_chars if total_chars else 0.0
    return row_acc, char_acc


def export_onnx(model, out_path: Path, device) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    dummy = torch.randn(1, 1, IMAGE_HEIGHT, IMAGE_WIDTH, device=device)
    torch.onnx.export(
        model,
        dummy,
        str(out_path),
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=12,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练整行数字 CRNN 并导出 ONNX")
    parser.add_argument("--labels", default="dataset/labels.csv",
                        help="labels.csv 路径")
    parser.add_argument("--out", default="models/row_crnn.onnx",
                        help="ONNX 输出路径")
    parser.add_argument("--epochs", type=int, default=40,
                        help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="批大小")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="学习率")
    parser.add_argument("--seed", type=int, default=2026,
                        help="随机种子")
    parser.add_argument("--device", default="cpu",
                        help="训练设备，默认 cpu")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    project_dir = Path.cwd()
    labels_path = Path(args.labels)
    if not labels_path.exists():
        print(f"[ERROR] 标注文件不存在: {labels_path}")
        return 1

    all_rows = load_label_rows(labels_path)
    if not all_rows:
        print(f"[ERROR] 标注文件为空: {labels_path}")
        return 1
    if len(all_rows) < 10:
        print("[WARN] 标注样本少于 10 条，只适合 smoke test，不适合正式训练")

    train_rows, val_rows, test_rows = load_or_create_splits(labels_path, all_rows, args.seed)
    if not train_rows:
        print("[ERROR] 训练集为空")
        return 1
    print(f"[INFO] train={len(train_rows)}, val={len(val_rows)}, test={len(test_rows)}")

    device = torch.device(args.device)
    model = RowCRNN().to(device)
    train_loader = DataLoader(
        RowDataset(train_rows, project_dir, augment=True),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_batch,
    )
    val_loader = DataLoader(
        RowDataset(val_rows, project_dir, augment=False),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_batch,
    )

    criterion = nn.CTCLoss(blank=BLANK_INDEX, zero_infinity=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_row_acc = -1.0
    best_state = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for images, targets, target_lengths, _ in train_loader:
            images = images.to(device)
            targets = targets.to(device)
            target_lengths = target_lengths.to(device)

            logits = model(images)
            log_probs = logits.log_softmax(dim=-1).permute(1, 0, 2)
            input_lengths = torch.full(
                size=(images.size(0),),
                fill_value=logits.size(1),
                dtype=torch.long,
                device=device,
            )

            loss = criterion(log_probs, targets, input_lengths, target_lengths)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())

        val_row_acc, val_char_acc = evaluate(model, val_loader, device)
        avg_loss = total_loss / max(1, len(train_loader))
        print(
            f"[EPOCH {epoch:03d}] loss={avg_loss:.4f}, "
            f"val_row_acc={val_row_acc:.4f}, val_char_acc={val_char_acc:.4f}"
        )

        if val_row_acc > best_row_acc:
            best_row_acc = val_row_acc
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    out_path = Path(args.out)
    export_onnx(model, out_path, device)
    torch.save(model.state_dict(), out_path.with_suffix(".pt"))

    if test_rows:
        test_loader = DataLoader(
            RowDataset(test_rows, project_dir, augment=False),
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_batch,
        )
        test_row_acc, test_char_acc = evaluate(model, test_loader, device)
        print(f"[TEST] row_acc={test_row_acc:.4f}, char_acc={test_char_acc:.4f}")

    print(f"[DONE] ONNX 已导出: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
