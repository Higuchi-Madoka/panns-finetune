# Finetune工程
finetune/文件夹是对原单文件脚本finetune_knock_detection_dist.py的重构，提供更清晰的代码结构。
transfer/inetune_knock_detection_dist.py是由PANNs开源代码库中的finetune_template.py改进得来
项目位于 @dragon03：/work105/wangzixu/project/bullying_scene_classification/。
## 工程结构

**`main.py`** (入口): 解析命令行参数，初始化环境，并组织训练循环。
**`dataset.py`** (数据): 负责音频文件加载。包含数据读取、清洗、以及数据增强的逻辑（可选）。
**`models.py`** (模型): 定义神经网络结构。包含 PANNs 的 `Cnn14` 骨干网和自定义的分类头（softmax层+二分类头），以及渐进式解冻策略。
**`trainer.py`** (训练): 定义了训练一个 Epoch的过程，和“如何验证”。包含具体的训练步进、梯度更新和阈值优化。
**`losses.py`** (损失): 定义损失函数。目前使用的是解决样本不平衡的 `FocalLoss`。
**`utils.py`** (工具): 包含分布式训练设置、画图、日志记录等辅助功能。

## 环境准备
### 0. 环境部署

1. 创建 conda 环境
mamba env create -f environment.yml

2. 激活
mamba activate panns_bsc

3. 安装 pip 依赖 
pip install -r requirements.txt

### 1. 数据集结构

dataset/
├── train/
│   ├── positive/ 
│   └── negative/
└── test/
    ├── positive/ ...
    └── negative/ ...

### 2. 运行训练

**单卡训练 (Mac MPS / 单 GPU / CPU):**
在项目根目录下运行：

# Mac OS
python -m finetune.main --mps --batch_size 8 --epochs 20

# NVIDIA GPU
python -m finetune.main --cuda --batch_size 16 --epochs 50

**多卡分布式训练 (DDP):**
# 例如：使用 4 张显卡
torchrun --nproc_per_node=4 -m finetune.main --cuda --batch_size 16 --epochs 100
* 多卡训练速度提升不大。主要目的是解决在batch_size > 16时，单卡训练可能会出现的显存益出。

## 主要类
* 修改数据增强策略：dataset.py 中的SimpleAudioAugmentation 类。
* 修改模型结构：修改models.py，中的 Transfer_Cnn14_Violence 类。
* 调整学习率或优化器：去 main.py，在 train 函数中找到 optimizer 和 scheduler 。
* 修改损失函数：在 losses.py 中修改，然后在 main.py 中调用。

## 训练策略

可启用以下策略
1.  **渐进式解冻**: 训练初期冻结预训练层，随着 Epoch 增加逐步解冻。逻辑在 models.py 中。
2.  **动态阈值优化**: 每个 Epoch 结束后，会自动在test上寻找最佳分类阈值（而不是固定的的 0.5）。具体逻辑在 trainer.py 中可以找到。
3.  **数据增强**: 数据量极少时可启用，在dataset.py中开启，可能需要debug。