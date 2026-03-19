"""
download_data.py - 自动下载 MNIST 数据集到 ./data 目录

用法：
    python download_data.py
"""

from torchvision import datasets, transforms

def main():
    print("正在下载 MNIST 数据集到 ./data ...")
    transform = transforms.ToTensor()
    datasets.MNIST("./data", train=True,  download=True, transform=transform)
    datasets.MNIST("./data", train=False, download=True, transform=transform)
    print("下载完成！数据保存在 ./data/MNIST/raw/")

if __name__ == "__main__":
    main()
