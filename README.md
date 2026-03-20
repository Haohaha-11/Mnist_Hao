# MNIST 手写数字识别

基于 PyTorch 的 MNIST 手写数字分类项目，使用卷积神经网络（CNN）实现，代码结构清晰，训练/验证/测试分离。

---

## 项目结构

```
Mnist_Hao/
├── data/                  # 本地数据集目录（需手动下载，见下方说明）
├── checkpoints/           # 模型权重保存目录
├── user_images/           # 放入自定义图像，供 test_demo.py 推理
├── Mnist_Original/        # PyTorch 官方原版 MNIST 示例（参考对比用）
├── model.py               # CNN 模型定义
├── dataset.py             # 数据加载与划分（训练/验证/测试）
├── train.py               # 训练 + 验证
├── test.py                # 测试集评估
├── test_demo.py           # 对 user_images/ 中的图像进行推理
├── download_data.py       # 数据集下载辅助脚本
├── requirements.txt       # 依赖列表
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

> 如官方地址访问受限，可使用国内镜像：
> https://mirror.coggle.club/dataset/fashion-mnist/

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

定义于 `model.py`，网络结构如下：

```
输入 (1, 28, 28)
  → Conv2d(1→32, 3×3) → ReLU
  → Conv2d(32→64, 3×3) → ReLU
  → MaxPool2d(2×2) → Dropout(0.25)
  → Flatten → Linear(9216→128) → ReLU → Dropout(0.5)
  → Linear(128→10) → LogSoftmax
```

---

## 训练与验证

```bash
# 基本训练（CPU）
python train.py

# 保存最优模型
python train.py --save-model

# 自定义参数示例
python train.py --epochs 20 --lr 0.5 --batch-size 128 --val-ratio 0.1 --save-model
```

### 训练参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--epochs` | 14 | 训练轮数 |
| `--lr` | 1.0 | 初始学习率（Adadelta 优化器） |
| `--gamma` | 0.7 | 学习率衰减系数（每 epoch × gamma） |
| `--batch-size` | 64 | 训练 batch size |
| `--val-ratio` | 0.1 | 验证集占训练集的比例 |
| `--seed` | 1 | 随机种子 |
| `--log-interval` | 10 | 每隔多少 batch 打印一次日志 |
| `--no-accel` | — | 禁用 GPU，强制使用 CPU |
| `--dry-run` | — | 单步快速验证流程是否跑通 |
| `--save-model` | — | 保存验证集最优模型到 `checkpoints/` |

训练结束后，模型权重保存在：

```
checkpoints/
├── best_model.pt   # 验证集准确率最高的模型
└── last_model.pt   # 最后一个 epoch 的模型
```

---

## 测试

```bash
# 使用默认路径 checkpoints/best_model.pt
python test.py

# 指定权重文件
python test.py --model-path checkpoints/last_model.pt
```

### 测试参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model-path` | `checkpoints/best_model.pt` | 模型权重路径 |
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
| `--model-path` | `checkpoints/best_model.pt` | 模型权重路径 |
| `--image-dir` | `user_images` | 图像文件夹路径 |
| `--no-accel` | — | 禁用 GPU，强制 CPU |

支持格式：`.png` `.jpg` `.jpeg` `.bmp` `.gif` `.tiff`

> 图像会自动转为灰度并 resize 到 28×28，预处理与训练时保持一致。

---

## 预期性能

在标准配置（14 epochs，lr=1.0，Adadelta）下，测试集准确率通常可达 **99%+**。

---

## 参考

- [PyTorch 官方 MNIST 示例](https://github.com/pytorch/examples/tree/main/mnist)
- [MNIST 数据集主页](http://yann.lecun.com/exdb/mnist/)
