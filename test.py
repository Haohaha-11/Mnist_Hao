"""
test.py - 在测试集上评估模型

用法：
    python test.py --model-path checkpoints/best_model.pt
"""

import argparse
import torch
import torch.nn.functional as F

from model import Net
from dataset import get_dataloaders


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction="sum").item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100.0 * correct / len(test_loader.dataset)
    print(f"\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n")
    return test_loss, accuracy


def main():
    parser = argparse.ArgumentParser(description="MNIST 测试")
    parser.add_argument("--model-path",     type=str, default="checkpoints/best_model.pt", help="模型权重路径")
    parser.add_argument("--test-batch-size", type=int, default=1000, help="测试 batch size (默认 1000)")
    parser.add_argument("--seed",           type=int, default=1,    help="随机种子 (默认 1)")
    parser.add_argument("--no-accel",       action="store_true",    help="禁用 GPU，强制 CPU")
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    use_accel = not args.no_accel and torch.accelerator.is_available()
    device = torch.accelerator.current_accelerator() if use_accel else torch.device("cpu")
    print(f"使用设备：{device}")

    _, _, test_loader = get_dataloaders(
        data_dir="./data",
        test_batch_size=args.test_batch_size,
        seed=args.seed,
    )

    model = Net().to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    print(f"已加载模型权重：{args.model_path}")

    test(model, device, test_loader)


if __name__ == "__main__":
    main()
