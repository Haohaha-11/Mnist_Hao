"""
model.py - MNIST CNN 模型定义

网络结构：
  输入 (1, 28, 28)
    → Conv2d(1→32, 3x3) → ReLU
    → Conv2d(32→64, 3x3) → ReLU
    → MaxPool2d(2x2) → Dropout(0.25)
    → Flatten
    → Linear(9216→128) → ReLU → Dropout(0.5)
    → Linear(128→10) → LogSoftmax
"""

import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = x.flatten(1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
