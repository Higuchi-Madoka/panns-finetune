# PANNs-Finetune
本项目是一个面向**单标签音频事件分类（默认二分类：positive/negative）**的 PANN微调脚本 。项目基于 PANNs  `finetune_template.py` 重构与扩展，在 `Cnn14` backbone 上增加可替换的分类头（含 softmax 输出），并提供更工程化的训练流程。

适用于小规模音频数据集（如某类事件 vs 非事件），希望快速把预训练 PANNs 迁移到你的任务上，并在 Mac MPS / CUDA / DDP 环境下复现训练与验证流程。

## 工程结构（在finetune目录下）

**`main.py`** (入口): 解析命令行参数，初始化环境，并组织训练循环。
**`dataset.py`** (数据): 负责音频文件加载。包含数据读取、清洗、以及数据增强的逻辑（可选）。
**`models.py`** (模型): 定义神经网络结构。包含 PANNs 的 `Cnn14` backbone和自定义的分类头（softmax层+二分类头），以及渐进式解冻策略。
**`trainer.py`** (训练): 定义了训练一个 Epoch的过程，和“如何验证”。包含具体的训练步进、梯度更新和阈值优化。
**`losses.py`** (损失): 定义损失函数。目前使用的是解决样本不平衡的 `FocalLoss`。
**`utils.py`** (工具): 包含分布式训练设置、画图、日志记录等辅助功能。

## 环境准备
### 0. 环境部署

1. 创建 conda 环境
```text
mamba env create -f environment.yml
```

2. 激活
```text
mamba activate panns_bsc
```

3. 安装 pip 依赖
```text
pip install -r requirements.txt
```
### 1. 数据集结构

```text
dataset/
├── train/
│   ├── positive/
│   └── negative/
└── test/
    ├── positive/
    └── negative/
```
### 2. 运行训练

**单卡训练 (Mac MPS / 单 GPU / CPU):**

**MacOS (MPS):**
```text
python -m finetune.main --mps --batch_size 8 --epochs 20
```
#### NVIDIA GPU (CUDA):
```text
python -m finetune.main --cuda --batch_size 16 --epochs 50
```
**多卡分布式训练 (DDP):**
例如使用 4 张显卡
```text
torchrun --nproc_per_node=4 -m finetune.main --cuda --batch_size 16 --epochs 100
```
* 多卡加速效果有限，但可以在“单卡放不下”的情况下仍能训练更大的全局 batch

## 主要类
* 修改数据增强策略：dataset.py 中的SimpleAudioAugmentation 类。
* 修改模型结构：修改models.py，中的 Transfer_Cnn14_Violence 类。
* 调整学习率或优化器：去 main.py，在 train 函数中找到 optimizer 和 scheduler 。
* 修改损失函数：在 losses.py 中修改，然后在 main.py 中调用。

## 训练策略
以下策略可选择性启用：
1.  **渐进式解冻**: 训练初期冻结预训练层，随着 Epoch 增加逐步解冻。逻辑在 models.py 中。
2.  **动态阈值优化**: 每个 Epoch 结束后，会自动在test上寻找最佳分类阈值（而不是固定的的 0.5）。具体逻辑在 trainer.py 中可以找到。
3.  **数据增强**: 数据量极少时可启用，在dataset.py中开启，可能需要debug。
