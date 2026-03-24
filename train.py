"""
train.py - 训练 + 验证

用法：
    python train.py [选项]

常用选项：
    --epochs 30
    --lr 1.0 #优化器官方推荐设置
    --batch-size 64
    --val-ratio 0.1
    --no-accel              # 强制使用 CPU

模型自动保存到 checkpoints/：
    best_model.pt   ← val loss 最小的那轮
    last_model.pt   ← 最后一轮
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from model_ours import NetOurs  
from dataset import get_dataloaders


# ──────────────────────────────────────────────
# 单 epoch 训练
# ──────────────────────────────────────────────
def train_one_epoch(model, device, train_loader, optimizer, epoch, dry_run=False):
    model.train()
    total_loss = 0.0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if dry_run:
            break

    avg_loss = total_loss / len(train_loader)
    return avg_loss


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
    parser = argparse.ArgumentParser(description="MNIST 训练 + 验证")
    parser.add_argument("--batch-size",     type=int,   default=64,   help="训练 batch size (默认 64)")
    parser.add_argument("--val-batch-size", type=int,   default=1000, help="验证 batch size (默认 1000)")
    parser.add_argument("--epochs",         type=int,   default=30,   help="训练轮数 (默认 30)")
    parser.add_argument("--lr",             type=float, default=1,    help="学习率 (默认 1.0)")
    parser.add_argument("--gamma",          type=float, default=0.7,  help="学习率衰减系数 (默认 0.7)")
    parser.add_argument("--val-ratio",      type=float, default=0.1,  help="验证集比例 (默认 0.1)")
    parser.add_argument("--seed",           type=int,   default=1,    help="随机种子 (默认 1)")
    parser.add_argument("--no-accel",       action="store_true",      help="禁用 GPU, 强制 CPU")
    parser.add_argument("--dry-run",        action="store_true",      help="单步快速验证流程")
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    use_accel = not args.no_accel and torch.accelerator.is_available()
    device = torch.accelerator.current_accelerator() if use_accel else torch.device("cpu")
    print(f"使用设备：{device}")

    pin_memory = use_accel
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

    best_val_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_avg_loss = train_one_epoch(model, device, train_loader, optimizer, epoch,
                                         dry_run=args.dry_run)
        val_loss, val_acc = validate(model, device, val_loader)
        scheduler.step()

        # 终端每 epoch 一行汇总
        print(f"Epoch {epoch:>3} | Train Avg Loss: {train_avg_loss:.6f} | Val Loss: {val_loss:.6f} | Val Acc: {val_acc:.2f}%")

        # 每轮都保存 last
        torch.save(model.state_dict(), "checkpoints/last_model_ours.pt")

        # val loss 更低时更新 best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "checkpoints/best_model_ours.pt")
            print(f"  -> 保存最优模型（Epoch {epoch}, Val Loss {best_val_loss:.6f}）")

    print(f"\n训练完成！模型已保存到 checkpoints/")
    print(f"  best_model_ours.pt : Val Loss 最小 ({best_val_loss:.6f})")
    print(f"  last_model_ours.pt : 最后一轮 (Epoch {args.epochs})")


if __name__ == "__main__":
    main()
