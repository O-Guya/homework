# 牙齿分割深度学习项目

## 项目概述

本项目实现了一个基于深度学习的牙齿分割算法，能够根据输入的RGB牙齿图像和特定的牙齿ID，输出该牙齿的像素级分割mask。

## 项目特点

- **条件化分割**: 使用FiLM (Feature-wise Linear Modulation) 机制将牙齿ID信息注入到U-Net模型中
- **多损失函数**: 结合BCE损失、Dice损失和Focal损失
- **完整评估**: 实现像素准确率、IoU、Dice系数等多种评估指标
- **可视化工具**: 提供丰富的可视化和分析工具
- **模块化设计**: 代码结构清晰，易于扩展和维护

## 项目结构

```
homework/
├── config.py                 # 配置文件
├── requirements.txt          # 依赖包列表
├── train.py                 # 训练脚本
├── test.py                  # 测试脚本
├── models/                  # 模型定义
│   ├── __init__.py
│   ├── unet.py             # U-Net模型with tooth ID conditioning
│   └── loss.py             # 损失函数
├── utils/                   # 工具函数
│   ├── __init__.py
│   ├── dataset.py          # 数据加载器
│   ├── metrics.py          # 评估指标
│   └── visualization.py    # 可视化工具
├── data/                    # 数据目录
│   └── ToothSegmDataset/
│       ├── trainset_valset/ # 训练和验证数据
│       └── testset/        # 测试数据
└── outputs/                 # 输出目录
    ├── models/             # 保存的模型
    ├── logs/               # 训练日志
    ├── results/            # 测试结果
    └── visualizations/     # 可视化结果
```

## 环境配置

### 安装依赖

```bash
pip install -r requirements.txt
```

### 主要依赖

- PyTorch >= 1.9.0
- torchvision >= 0.10.0
- OpenCV >= 4.5.0
- matplotlib >= 3.4.0
- albumentations >= 1.1.0
- scikit-learn >= 1.0.0

## 数据格式

### 数据集结构

- **训练数据**: `trainset_valset/` - 按牙齿ID组织，每个ID包含多个患者的RGB图像和对应的mask图像
- **测试数据**: `testset/` - 只包含RGB图像，用于最终测试

### 牙齿ID映射

```python
TOOTH_ID_MAPPING = {
    "11": 0, "12": 1, "13": 2, "14": 3, "15": 4, "16": 5, "17": 6,
    "21": 8, "22": 9, "23": 10, "24": 11, "25": 12, "26": 13, "27": 14,
    "34": 19, "35": 20, "36": 21, "37": 22,
    "45": 28, "46": 29, "47": 30
}
```

### Mask编码

- Value 1: 上牙龈
- Value 2: 下牙龈  
- Value 3: 目标牙齿

## 使用方法

### 1. 训练模型

```bash
# 使用默认配置训练
python train.py

# 自定义参数训练
python train.py --batch_size 16 --epochs 50 --lr 0.001 --tooth_ids 0 1 2 3

# 恢复训练
python train.py --resume outputs/models/best_model.pth
```

### 2. 测试模型

```bash
# 测试所有选择的牙齿ID
python test.py --model_path outputs/models/best_model.pth

# 测试特定牙齿ID
python test.py --model_path outputs/models/best_model.pth --tooth_ids 0 1 2 3 --num_samples 5
```

### 3. 推理接口

```python
from test import inference
import cv2
import numpy as np

# 加载图像
image = cv2.imread('path/to/image.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 推理
mask = inference(
    input_image=image,
    tooth_id=0,  # 目标牙齿ID
    model_path='outputs/models/best_model.pth'
)

# 显示结果
from utils.visualization import show_anns
show_anns(image, mask, title="分割结果")
```

## 模型架构

### U-Net with Tooth ID Conditioning

模型采用U-Net架构，通过以下方式实现牙齿ID条件化：

1. **牙齿ID嵌入**: 将牙齿ID转换为高维嵌入向量
2. **FiLM调制**: 使用Feature-wise Linear Modulation将牙齿ID信息注入到特征中
3. **多尺度条件化**: 在编码器的多个层级应用条件化

### 关键组件

- **编码器**: 4层下采样，提取多尺度特征
- **瓶颈层**: 最深层特征提取
- **解码器**: 4层上采样，恢复空间分辨率
- **FiLM层**: 在每层应用条件化调制
- **跳跃连接**: 保留细节信息

## 损失函数

### 组合损失

```python
Loss = BCE_weight * BCE_Loss + Dice_weight * Dice_Loss + Focal_weight * Focal_Loss
```

- **BCE损失**: 二元交叉熵损失
- **Dice损失**: 处理类别不平衡
- **Focal损失**: 关注难分类样本

## 评估指标

- **像素准确率 (Pixel Accuracy)**: 正确分类的像素比例
- **平均像素准确率 (mPA)**: 各类别像素准确率的平均值
- **IoU**: 交集与并集的比值
- **平均IoU (mIoU)**: 各类别IoU的平均值
- **Dice系数**: 2 * 交集 / (预测 + 真实)

## 训练策略

### 数据增强

- 水平/垂直翻转
- 旋转 (±15度)
- 亮度/对比度调整
- 随机缩放

### 优化策略

- **优化器**: AdamW
- **学习率调度**: ReduceLROnPlateau
- **早停**: 验证损失不再下降时停止
- **权重衰减**: L2正则化

### 超参数

```python
TRAIN_CONFIG = {
    "batch_size": 8,
    "num_epochs": 100,
    "learning_rate": 1e-4,
    "weight_decay": 1e-5,
    "patience": 15,
    "val_split": 0.1,
    "image_size": (256, 256)
}
```

## 可视化功能

### 训练过程可视化

- 训练/验证损失曲线
- 学习率调度曲线
- 指标变化趋势

### 结果可视化

- 原始图像、真实mask、预测mask对比
- 叠加显示
- 差异分析
- 不同牙齿ID性能对比

### 调试工具

- 注意力可视化
- 损失景观分析
- 数据样本展示

## 实验记录

### 训练监控

使用TensorBoard记录训练过程：

```bash
tensorboard --logdir outputs/logs
```

### 结果保存

- **模型检查点**: `outputs/models/`
- **训练日志**: `outputs/logs/`
- **测试结果**: `outputs/results/`
- **可视化图像**: `outputs/visualizations/`

## 性能优化

### 内存优化

- 使用混合精度训练
- 梯度累积
- 数据加载优化

### 计算优化

- GPU加速
- 批处理推理
- 模型量化

## 故障排除

### 常见问题

1. **CUDA内存不足**: 减小batch_size
2. **数据加载错误**: 检查数据路径和格式
3. **模型收敛慢**: 调整学习率或损失函数权重

### 调试建议

1. 使用小数据集测试
2. 检查数据预处理
3. 监控梯度范数
4. 可视化中间结果

## 扩展功能

### 模型改进

- 添加注意力机制
- 使用更深的网络
- 多尺度训练

### 数据增强

- 弹性变形
- 颜色空间变换
- 混合增强

### 后处理

- 形态学操作
- 连通域分析
- 轮廓优化

## 贡献指南

1. Fork项目
2. 创建功能分支
3. 提交更改
4. 创建Pull Request

## 许可证

本项目采用MIT许可证。

## 联系方式

如有问题或建议，请通过以下方式联系：

- 邮箱: [your-email@example.com]
- GitHub: [your-github-username]

## 致谢

感谢所有为这个项目做出贡献的开发者和研究人员。
