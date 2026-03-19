"""
dataset.py - MNIST 数据集加载与划分

使用说明：
  - 数据需提前下载到 ./data 目录（见 README.md）
  - 训练集 60000 张，从中划分出 val_ratio 比例作为验证集
  - 测试集 10000 张，独立使用
"""

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

# 官方统计的 MNIST 均值与标准差
MEAN = (0.1307,)
STD  = (0.3081,)


def get_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def get_dataloaders(
    data_dir: str = "./data",
    batch_size: int = 64,
    test_batch_size: int = 1000,
    val_ratio: float = 0.1,
    seed: int = 1,
    num_workers: int = 2,
    pin_memory: bool = False,
):
    """
    返回 train_loader, val_loader, test_loader

    参数：
        data_dir       : 本地数据目录（需已下载，download=False）
        batch_size     : 训练/验证 batch size
        test_batch_size: 测试 batch size
        val_ratio      : 从训练集中划出的验证集比例，默认 10%
        seed           : 随机种子，保证划分可复现
        num_workers    : DataLoader 工作进程数
        pin_memory     : 是否锁页内存（GPU 训练时建议 True）
    """
    transform = get_transform()

    # 原始训练集 60000 张（不自动下载，需本地已有数据）
    full_train = datasets.MNIST(data_dir, train=True, download=False, transform=transform)
    # 测试集 10000 张
    test_set   = datasets.MNIST(data_dir, train=False, download=False, transform=transform)

    # 按比例划分训练集 / 验证集
    val_size   = int(len(full_train) * val_ratio)
    train_size = len(full_train) - val_size
    generator  = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(full_train, [train_size, val_size], generator=generator)

    common_kwargs = dict(num_workers=num_workers, pin_memory=pin_memory)

    train_loader = DataLoader(train_set, batch_size=batch_size,     shuffle=True,  **common_kwargs)
    val_loader   = DataLoader(val_set,   batch_size=test_batch_size, shuffle=False, **common_kwargs)
    test_loader  = DataLoader(test_set,  batch_size=test_batch_size, shuffle=False, **common_kwargs)

    print(f"数据划分完成：训练集 {train_size} | 验证集 {val_size} | 测试集 {len(test_set)}")
    return train_loader, val_loader, test_loader
