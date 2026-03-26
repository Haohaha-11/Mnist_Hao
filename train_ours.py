"""
train_ours.py - 训练 + 验证（NetOurs / model_ours.py）

用法：
    python train_ours.py [选项]

常用选项：
    --epochs 30
    --lr 1.0
    --batch-size 64
    --val-ratio 0.1
    --no-accel              # 强制使用 CPU

模型自动保存到 checkpoints/：
    best_model_ours.pt   ← val loss 最小的那轮
    last_model_ours.pt   ← 最后一轮
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
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

from model_ours import NetOurs
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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FONT_PATH = os.path.join(BASE_DIR, "Comic-Sans-MS-Regular-2.ttf")


def _font(size=12):
    if os.path.exists(FONT_PATH):
        return fm.FontProperties(fname=FONT_PATH, size=size)
    return fm.FontProperties(size=size)


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
# 绘图并保存
# ──────────────────────────────────────────────
def plot_metrics(epochs_list, train_losses, val_losses, train_accs, val_accs, save_path):
    fp_title  = _font(15)
    fp_label  = _font(12)
    fp_tick   = _font(10)
    fp_legend = _font(11)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#F8F9FA")

    # ── Loss ──
    ax1.set_facecolor("#FFFFFF")
    ax1.plot(epochs_list, train_losses, color="#2196F3", linewidth=2.2,
             marker="o", markersize=4, label="Train Loss")
    ax1.plot(epochs_list, val_losses,   color="#F44336", linewidth=2.2,
             marker="s", markersize=4, label="Val Loss")
    ax1.set_title("Loss Curve", fontproperties=fp_title, pad=12)
    ax1.set_xlabel("Epoch",     fontproperties=fp_label)
    ax1.set_ylabel("Loss",      fontproperties=fp_label)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.spines[["top", "right"]].set_visible(False)
    for lbl in ax1.get_xticklabels() + ax1.get_yticklabels():
        lbl.set_fontproperties(fp_tick)
    ax1.legend(prop=fp_legend, framealpha=0.8)

    # ── Accuracy ──
    ax2.set_facecolor("#FFFFFF")
    ax2.plot(epochs_list, train_accs, color="#4CAF50", linewidth=2.2,
             marker="o", markersize=4, label="Train Acc")
    ax2.plot(epochs_list, val_accs,   color="#FF9800", linewidth=2.2,
             marker="s", markersize=4, label="Val Acc")
    ax2.set_title("Accuracy Curve",  fontproperties=fp_title, pad=12)
    ax2.set_xlabel("Epoch",           fontproperties=fp_label)
    ax2.set_ylabel("Accuracy (%)",    fontproperties=fp_label)
    ax2.set_ylim(0, 105)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.spines[["top", "right"]].set_visible(False)
    for lbl in ax2.get_xticklabels() + ax2.get_yticklabels():
        lbl.set_fontproperties(fp_tick)
    ax2.legend(prop=fp_legend, framealpha=0.8)

    plt.suptitle("NetOurs  —  Training Metrics", fontproperties=_font(17), y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"训练曲线已保存：{save_path}")


# ──────────────────────────────────────────────
# 主流程
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="MNIST 训练 + 验证（NetOurs）")
    parser.add_argument("--batch-size",     type=int,   default=64,   help="训练 batch size (默认 64)")
    parser.add_argument("--val-batch-size", type=int,   default=1000, help="验证 batch size (默认 1000)")
    parser.add_argument("--epochs",         type=int,   default=30,   help="训练轮数 (默认 30)")
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
    print(f"使用设备：{device}")

    pin_memory = device.type == "cuda"
    train_loader, val_loader, _ = get_dataloaders(
        data_dir="./data",
        batch_size=args.batch_size,
        test_batch_size=args.val_batch_size,
        val_ratio=args.val_ratio,
        seed=args.seed,
        pin_memory=pin_memory,
    )

    model = NetOurs().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    os.makedirs("checkpoints", exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = f"checkpoints/train_ours_log_{timestamp}.txt"
    tee = Tee(log_path)
    print(f"日志保存至：{log_path}")
    print(f"使用设备：{device}")
    print(f"模型：NetOurs | Epochs: {args.epochs} | LR: {args.lr} | Batch: {args.batch_size}")
    print("-" * 75)

    best_val_loss = float("inf")
    epochs_list, train_losses, val_losses, train_accs, val_accs = [], [], [], [], []

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, device, train_loader, optimizer,
                                                dry_run=args.dry_run)
        val_loss, val_acc = validate(model, device, val_loader)
        scheduler.step()

        epochs_list.append(epoch)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(f"Epoch {epoch:>3} | Train Loss: {train_loss:.6f} | Train Acc: {train_acc:.2f}% "
              f"| Val Loss: {val_loss:.6f} | Val Acc: {val_acc:.2f}%")

        torch.save(model.state_dict(), "checkpoints/last_model_ours_325.pt")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "checkpoints/best_model_ours_325.pt")
            print(f"  -> 保存最优模型（Epoch {epoch}, Val Loss {best_val_loss:.6f}）")

        if args.dry_run:
            break

    print(f"\n训练完成！模型已保存到 checkpoints/")
    print(f"  best_model_ours_325.pt : Val Loss 最小 ({best_val_loss:.6f})")
    print(f"  last_model_ours_325.pt : 最后一轮 (Epoch {args.epochs})")

    plot_metrics(epochs_list, train_losses, val_losses, train_accs, val_accs,
                 "checkpoints/training_curve_ours.png")
    tee.close()


if __name__ == "__main__":
    main()
