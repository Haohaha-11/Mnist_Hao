"""
train.py - 训练 + 验证

用法：
    python train.py [选项]

常用选项：
    --epochs 14
    --lr 1.0
    --batch-size 64
    --val-ratio 0.1
    --save-model            # 保存最优模型到 checkpoints/
    --no-accel              # 强制使用 CPU
"""

import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from model import Net
from dataset import get_dataloaders


# ──────────────────────────────────────────────
# 单 epoch 训练
# ──────────────────────────────────────────────
def train_one_epoch(model, device, train_loader, optimizer, epoch, log_interval=10, dry_run=False):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % log_interval == 0:
            print(
                f"Train Epoch: {epoch} "
                f"[{batch_idx * len(data)}/{len(train_loader.dataset)} "
                f"({100. * batch_idx / len(train_loader):.0f}%)]\t"
                f"Loss: {loss.item():.6f}"
            )
            if dry_run:
                break


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
    print(f"\nVal set:  Average loss: {val_loss:.4f}, Accuracy: {correct}/{len(val_loader.dataset)} ({accuracy:.2f}%)\n")
    return val_loss, accuracy


# ──────────────────────────────────────────────
# 主流程
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="MNIST 训练 + 验证")
    parser.add_argument("--batch-size",   type=int,   default=64,   help="训练 batch size (默认 64)")
    parser.add_argument("--val-batch-size", type=int, default=1000, help="验证 batch size (默认 1000)")
    parser.add_argument("--epochs",       type=int,   default=14,   help="训练轮数 (默认 14)")
    parser.add_argument("--lr",           type=float, default=1.0,  help="学习率 (默认 1.0)")
    parser.add_argument("--gamma",        type=float, default=0.7,  help="学习率衰减系数 (默认 0.7)")
    parser.add_argument("--val-ratio",    type=float, default=0.1,  help="验证集比例 (默认 0.1)")
    parser.add_argument("--seed",         type=int,   default=1,    help="随机种子 (默认 1)")
    parser.add_argument("--log-interval", type=int,   default=10,   help="日志打印间隔 batch 数")
    parser.add_argument("--no-accel",     action="store_true",      help="禁用 GPU，强制 CPU")
    parser.add_argument("--dry-run",      action="store_true",      help="单步快速验证流程")
    parser.add_argument("--save-model",   action="store_true",      help="保存最优模型权重")
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

    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        train_one_epoch(model, device, train_loader, optimizer, epoch,
                        log_interval=args.log_interval, dry_run=args.dry_run)
        val_loss, val_acc = validate(model, device, val_loader)
        scheduler.step()

        if args.save_model and val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "checkpoints/best_model.pt")
            print(f"  -> 保存最优模型（验证准确率 {best_acc:.2f}%）")

    if args.save_model:
        torch.save(model.state_dict(), "checkpoints/last_model.pt")
        print("已保存最终模型到 checkpoints/last_model.pt")


if __name__ == "__main__":
    main()
