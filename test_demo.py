"""
test_demo.py - 对 user_images/ 文件夹中的图像进行推理

用法：
    python test_demo.py
    python test_demo.py --model-path checkpoints/best_model.pt --image-dir user_images
"""

import argparse
import os
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from model import Net

SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff"}

# 与 MNIST 训练时一致的预处理
preprocess = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])


def predict_image(model, device, img_path: Path) -> int:
    img = Image.open(img_path).convert("RGB")
    tensor = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(tensor)
    return output.argmax(dim=1).item()


def main():
    parser = argparse.ArgumentParser(description="对 user_images/ 中的图像进行 MNIST 数字识别")
    parser.add_argument("--model-path", type=str, default="checkpoints/best_model.pt", help="模型权重路径")
    parser.add_argument("--image-dir",  type=str, default="user_images",               help="用户图像文件夹")
    parser.add_argument("--no-accel",   action="store_true",                            help="禁用 GPU，强制 CPU")
    args = parser.parse_args()

    # 设备
    use_accel = not args.no_accel and torch.accelerator.is_available()
    device = torch.accelerator.current_accelerator() if use_accel else torch.device("cpu")
    print(f"使用设备：{device}")

    # 加载模型
    if not os.path.exists(args.model_path):
        print(f"[错误] 找不到模型文件：{args.model_path}")
        print("请先运行 train.py 训练模型，或通过 --model-path 指定正确路径。")
        return

    model = Net().to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    print(f"已加载模型：{args.model_path}\n")

    # 检查图像文件夹
    image_dir = Path(args.image_dir)
    if not image_dir.exists():
        image_dir.mkdir(parents=True)
        print(f"已创建文件夹 {image_dir}，请将图像放入后重新运行。")
        return

    images = [p for p in sorted(image_dir.iterdir()) if p.suffix.lower() in SUPPORTED_EXTS]
    if not images:
        print(f"[提示] {image_dir} 中没有找到图像文件（支持格式：{', '.join(SUPPORTED_EXTS)}）")
        return

    print(f"共找到 {len(images)} 张图像，开始推理...\n")
    print(f"{'文件名':<30} {'预测数字':>6}")
    print("-" * 38)

    for img_path in images:
        try:
            pred = predict_image(model, device, img_path)
            print(f"{img_path.name:<30} {pred:>6}")
        except Exception as e:
            print(f"{img_path.name:<30} [错误] {e}")

    print("\n推理完成。")


if __name__ == "__main__":
    main()
