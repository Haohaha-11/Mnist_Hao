"""
model_ours.py - 改进版 MNIST CNN

改进点（相比 model.py）：
  多尺度特征融合（MultiScaleBlock）
    同一层用 1x1 和 3x3 两种卷积并行提取特征，再 concat 拼接
    → 同时捕捉局部细节和更广泛的上下文信息

网络结构：
  输入 (1, 28, 28)
    → MultiScaleBlock(1  → 32) : [1x1(16) ‖ 3x3(16)] → concat
    → MultiScaleBlock(32 → 64) : [1x1(32) ‖ 3x3(32)] → concat
    → MaxPool2d(2x2) → Dropout(0.25)
    → Flatten
    → Linear(12544 → 128) → ReLU → Dropout(0.5)
    → Linear(128 → 10) → LogSoftmax
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────
# 多尺度卷积块：1x1 和 3x3 并行，输出 concat 拼接
# ──────────────────────────────────────────────
class MultiScaleBlock(nn.Module):
    """
    并行使用两种卷积核提取特征，然后在通道维度拼接：
      - branch_1x1: 1x1 conv，捕捉逐点特征（out_channels // 2 个）
      - branch_3x3: 3x3 conv + padding=1 保持空间尺寸（out_channels // 2 个）
    拼接后共 out_channels 个通道。
    要求 out_channels 为偶数。
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        half = out_channels // 2
        self.branch_1x1 = nn.Conv2d(in_channels, half, kernel_size=1)
        self.branch_3x3 = nn.Conv2d(in_channels, half, kernel_size=3, padding=1)

    def forward(self, x):
        b1 = F.relu(self.branch_1x1(x))
        b2 = F.relu(self.branch_3x3(x))
        return torch.cat([b1, b2], dim=1)   # (B, out_channels, H, W)


# ──────────────────────────────────────────────
# 完整模型
# ──────────────────────────────────────────────
class NetOurs(nn.Module):
    def __init__(self):
        super().__init__()
        self.ms1      = MultiScaleBlock(1,  32)    # (B,  1, 28, 28) → (B, 32, 28, 28)
        self.ms2      = MultiScaleBlock(32, 64)    # (B, 32, 28, 28) → (B, 64, 28, 28)
        self.dropout1 = nn.Dropout(0.25)
        self.fc1      = nn.Linear(12544, 128)      # 64 * 14 * 14 = 12544（MaxPool 后）
        self.dropout2 = nn.Dropout(0.5)
        self.fc2      = nn.Linear(128, 10)

    def forward(self, x):
        x = self.ms1(x)                            # (B,  1, 28, 28) → (B, 32, 28, 28)
        x = self.ms2(x)                            # (B, 32, 28, 28) → (B, 64, 28, 28)
        x = F.max_pool2d(x, 2)                    # (B, 64, 28, 28) → (B, 64, 14, 14)
        x = self.dropout1(x)
        x = x.flatten(1)                           # (B, 12544)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
