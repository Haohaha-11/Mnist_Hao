# MNIST 手写数字识别

基于 PyTorch 的 MNIST 手写数字分类项目，提供**两套模型**：原版 CNN（`model.py`）与改进版多尺度 CNN（`model_ours.py`），代码结构清晰，训练/验证/测试分离。

---

## 项目结构

```
Mnist_Hao/
├── data/                  # 本地数据集目录（需手动下载，见下方说明）
├── checkpoints/           # 模型权重保存目录
├── user_images/           # 放入自定义图像，供 test_demo.py 推理
├── logs/                  # 训练日志目录
├── Mnist_Original/        # PyTorch 官方原版 MNIST 示例（参考对比用）
├── model.py               # 原版 CNN 模型定义
├── model_ours.py          # 改进版多尺度 CNN 模型定义
├── dataset.py             # 数据加载与划分（训练/验证/测试）
├── train.py               # 训练 + 验证（使用改进版模型）
├── test.py                # 测试集评估
├── test_demo.py           # 对 user_images/ 中的图像进行推理
├── download_data.py       # 数据集下载辅助脚本
├── requirements.txt       # 依赖列表
├── Recording_Hao.md       # 实验记录与改进笔记
└── README.md
```

---

## 数据集下载

MNIST 数据集需手动下载后放入 `./data` 目录，**不会自动联网下载**。

### 官方镜像（推荐）

| 文件 | 说明 | 下载地址 |
|------|------|----------|
| `train-images-idx3-ubyte.gz` | 训练图像 (60000 张) | http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz |
| `train-labels-idx1-ubyte.gz` | 训练标签 | http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz |
| `t10k-images-idx3-ubyte.gz`  | 测试图像 (10000 张) | http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz  |
| `t10k-labels-idx1-ubyte.gz`  | 测试标签 | http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz  |

### 存放结构

下载并解压后，`data/` 目录结构应如下：

```
data/
└── MNIST/
    └── raw/
        ├── train-images-idx3-ubyte
        ├── train-labels-idx1-ubyte
        ├── t10k-images-idx3-ubyte
        └── t10k-labels-idx1-ubyte
```

> torchvision 会自动识别 `data/MNIST/raw/` 下的原始文件，无需额外处理。

---

## 环境安装

```bash
pip install -r requirements.txt
```

如需 GPU 支持，请根据 CUDA 版本安装对应的 PyTorch，参考：https://pytorch.org/get-started/locally/

---

## 数据划分

| 子集 | 来源 | 数量 |
|------|------|------|
| 训练集 | MNIST train split 的 90% | 54000 |
| 验证集 | MNIST train split 的 10% | 6000  |
| 测试集 | MNIST test split（独立）  | 10000 |

划分逻辑见 `dataset.py`，通过 `val_ratio` 参数可调整比例，`seed` 保证可复现。

---

## 模型结构

### 原版 CNN（`model.py`）

```
输入 (1, 28, 28)
  → Conv2d(1→32, 3×3) → ReLU
  → Conv2d(32→64, 3×3) → ReLU
  → MaxPool2d(2×2) → Dropout(0.25)
  → Flatten → Linear(9216→128) → ReLU → Dropout(0.5)
  → Linear(128→10) → LogSoftmax
```

### 改进版多尺度 CNN（`model_ours.py`）

在原版基础上引入 **MultiScaleBlock**，每层并行使用 1×1 和 3×3 两种卷积核，concat 拼接后输出，同时捕捉局部细节与更广泛的上下文信息。

```
输入 (1, 28, 28)
  → MultiScaleBlock(1→32)  : [1×1(16) ‖ 3×3(16)] → concat → (B, 32, 28, 28)
  → MultiScaleBlock(32→64) : [1×1(32) ‖ 3×3(32)] → concat → (B, 64, 28, 28)
  → MaxPool2d(2×2) → Dropout(0.25)
  → Flatten → Linear(12544→128) → ReLU → Dropout(0.5)
  → Linear(128→10) → LogSoftmax
```

---

## 训练与验证

`train.py` 默认使用改进版模型（`NetOurs`）进行训练。

```bash
# 基本训练（自动检测 GPU）
python train.py

# 自定义参数示例
python train.py --epochs 30 --lr 1.0 --batch-size 64 --val-ratio 0.1

# 强制使用 CPU
python train.py --no-accel

# 单步快速验证流程是否跑通
python train.py --dry-run
```

### 训练参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--epochs` | 30 | 训练轮数 |
| `--lr` | 1.0 | 初始学习率（Adadelta 优化器） |
| `--gamma` | 0.7 | 学习率衰减系数（每 epoch × gamma） |
| `--batch-size` | 64 | 训练 batch size |
| `--val-ratio` | 0.1 | 验证集占训练集的比例 |
| `--seed` | 1 | 随机种子 |
| `--no-accel` | — | 禁用 GPU，强制使用 CPU |
| `--dry-run` | — | 单步快速验证流程是否跑通 |

训练结束后，模型权重保存在：

```
checkpoints/
├── best_model_ours.pt   # 验证集 loss 最小的模型（改进版）
└── last_model_ours.pt   # 最后一个 epoch 的模型（改进版）
```

---

## 测试

```bash
# 使用默认路径 checkpoints/best_model_ours.pt
python test.py

# 指定权重文件
python test.py --model-path checkpoints/last_model_ours.pt
```

### 测试参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model-path` | `checkpoints/best_model_ours.pt` | 模型权重路径 |
| `--test-batch-size` | 1000 | 测试 batch size |
| `--no-accel` | — | 禁用 GPU |

---

## 自定义图像推理（test_demo.py）

将自己的手写数字图像放入 `user_images/` 文件夹，然后运行：

```bash
python test_demo.py
```

脚本会自动对文件夹内所有图像逐一推理并打印预测结果。

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model-path` | `checkpoints/best_model_ours.pt` | 模型权重路径 |
| `--image-dir` | `user_images` | 图像文件夹路径 |
| `--no-accel` | — | 禁用 GPU，强制 CPU |

支持格式：`.png` `.jpg` `.jpeg` `.bmp` `.gif` `.tiff`

> 图像会自动转为灰度并 resize 到 28×28，预处理与训练时保持一致。

---

## 预期性能

| 模型 | 配置 | 测试集准确率 |
|------|------|-------------|
| 原版 CNN (`model.py`) | 14 epochs, lr=1.0, Adadelta | **99.16%** |
| 改进版 CNN (`model_ours.py`) | 30 epochs, lr=1.0, Adadelta | **99%+** |

---

## 参考

- [PyTorch 官方 MNIST 示例](https://github.com/pytorch/examples/tree/main/mnist)
- [MNIST 数据集主页](http://yann.lecun.com/exdb/mnist/)
