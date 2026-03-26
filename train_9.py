"""
train_9.py - 训练 + 验证（Net / model.py）

用法：
    python train_9.py [选项]

常用选项：
    --epochs 20
    --lr 1.0
    --batch-size 64
    --val-ratio 0.1
    --no-accel              # 强制使用 CPU

模型自动保存到 checkpoints/：
    best_model_9.pt   ← val loss 最小的那轮
    last_model_9.pt   ← 最后一轮
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import sys
import datetime
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from model import Net
from dataset import get_dataloaders


class Tee:
    """同时写入终端和文件"""
    def __init__(self, filepath):
        self._file = open(filepath, "w", encoding="utf-8")
        self._stdout = sys.stdout
        sys.stdout = self

    def write(self, data):
        self._stdout.write(data)
        self._file.write(data)

    def flush(self):
        self._stdout.flush()
        self._file.flush()

    def close(self):
        sys.stdout = self._stdout
        self._file.close()


# ──────────────────────────────────────────────
# 单 epoch 训练（返回 avg_loss, accuracy）
# ──────────────────────────────────────────────
def train_one_epoch(model, device, train_loader, optimizer, dry_run=False):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)

        if dry_run:
            break

    avg_loss = total_loss / len(train_loader)
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


# ──────────────────────────────────────────────
# 验证
# ──────────────────────────────────────────────
def validate(model, device, val_loader):
    model.eval()
    val_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss += F.nll_loss(output, target, reduction="sum").item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    val_loss /= len(val_loader.dataset)
    accuracy = 100.0 * correct / len(val_loader.dataset)
    return val_loss, accuracy


# ──────────────────────────────────────────────
# 主流程
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="MNIST 训练 + 验证（Net）")
    parser.add_argument("--batch-size",     type=int,   default=64,   help="训练 batch size (默认 64)")
    parser.add_argument("--val-batch-size", type=int,   default=1000, help="验证 batch size (默认 1000)")
    parser.add_argument("--epochs",         type=int,   default=20,   help="训练轮数 (默认 20)")
    parser.add_argument("--lr",             type=float, default=1.0,  help="学习率 (默认 1.0)")
    parser.add_argument("--gamma",          type=float, default=0.7,  help="学习率衰减系数 (默认 0.7)")
    parser.add_argument("--val-ratio",      type=float, default=0.1,  help="验证集比例 (默认 0.1)")
    parser.add_argument("--seed",           type=int,   default=1,    help="随机种子 (默认 1)")
    parser.add_argument("--no-accel",       action="store_true",      help="禁用 GPU，强制 CPU")
    parser.add_argument("--dry-run",        action="store_true",      help="单步快速验证流程")
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    if not args.no_accel and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    os.makedirs("checkpoints", exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = f"checkpoints/train_9_log_{timestamp}.txt"
    tee = Tee(log_path)
    print(f"日志保存至：{log_path}")
    print(f"使用设备：{device}")
    print(f"模型：Net | Epochs: {args.epochs} | LR: {args.lr} | Batch: {args.batch_size}")
    print("-" * 75)

    pin_memory = device.type == "cuda"
    train_loader, val_loader, _ = get_dataloaders(
        data_dir="./data",
        batch_size=args.batch_size,
        test_batch_size=args.val_batch_size,
        val_ratio=args.val_ratio,
        seed=args.seed,
        pin_memory=pin_memory,
    )

    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, device, train_loader, optimizer,
                                                dry_run=args.dry_run)
        val_loss, val_acc = validate(model, device, val_loader)
        scheduler.step()

        print(f"Epoch {epoch:>3} | Train Loss: {train_loss:.6f} | Train Acc: {train_acc:.2f}% "
              f"| Val Loss: {val_loss:.6f} | Val Acc: {val_acc:.2f}%")

        torch.save(model.state_dict(), "checkpoints/last_model_9.pt")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "checkpoints/best_model_9.pt")
            print(f"  -> 保存最优模型（Epoch {epoch}, Val Loss {best_val_loss:.6f}）")

        if args.dry_run:
            break

    print(f"\n训练完成！模型已保存到 checkpoints/")
    print(f"  best_model_9.pt : Val Loss 最小 ({best_val_loss:.6f})")
    print(f"  last_model_9.pt : 最后一轮 (Epoch {args.epochs})")
    tee.close()


if __name__ == "__main__":
    main()
