# 牙齿分割项目

这是一个基于深度学习的牙齿分割项目，使用U-Net架构进行医学图像分割。

## 项目结构

```
assignment1/
├── data/                          # 数据目录
│   └── ToothSegmDataset/         # 牙齿分割数据集
├── models/                        # 模型定义
│   ├── unet.py                   # U-Net模型
│   ├── loss.py                   # 损失函数
│   ├── optimizer.py              # 优化器
│   └── test_model.py             # 模型测试
├── utils/                         # 工具函数
│   ├── metrics.py                # 评估指标
│   └── visualization.py          # 可视化工具
├── results/                       # 结果输出
├── data_loader.py                # 数据加载器
├── train.py                      # 训练脚本
├── test_data_loader.py           # 数据加载器测试
├── config.py                     # 配置文件
└── requirements.txt              # 依赖包
```

## 数据集

数据集包含以下结构：
- `testset/`: 测试集，包含RGB图像
- `trainset_valset/`: 训练和验证集，包含RGB图像和对应的掩码

每个牙齿类型都有独立的文件夹，包含多个患者的图像。

## 特性

### 健壮的数据加载
- 自动处理损坏的图像文件
- 支持多种图像格式
- 数据增强和预处理
- 跳过无法加载的样本

### 模型架构
- U-Net分割网络
- 支持多类别分割
- 可配置的网络深度和宽度

### 训练功能
- 自动混合精度训练
- 学习率调度
- 早停机制
- 模型检查点保存
- TensorBoard日志记录

### 评估指标
- Dice分数
- IoU分数
- 像素准确率
- 精确率、召回率、F1分数

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 1. 测试数据加载器

```bash
python test_data_loader.py
```

### 2. 训练模型

```bash
python train.py
```

### 3. 配置参数

修改 `config.py` 文件来调整训练参数：

```python
MODEL_CONFIG = {
    'num_classes': 2,
    'image_size': 512,
    'batch_size': 8,
    'epochs': 100,
    'learning_rate': 1e-4,
    # ... 其他参数
}
```

## 数据加载器特性

### 处理损坏图像
数据加载器能够自动处理以下问题：
- 损坏的JPEG文件
- 无法识别的图像格式
- 缺失的掩码文件
- 加载失败的图像

### 数据增强
- 随机水平/垂直翻转
- 随机旋转
- 亮度/对比度调整
- 噪声添加（可选）

### 内存优化
- 按需加载图像
- 支持多进程数据加载
- 自动跳过损坏文件

## 模型训练

### 损失函数
- Dice损失：处理类别不平衡
- 交叉熵损失：辅助训练
- 组合损失：Dice + 0.5 * CrossEntropy

### 优化器
- Adam优化器
- 学习率调度
- 权重衰减

### 监控
- 实时训练进度
- TensorBoard可视化
- 自动保存最佳模型

## 结果输出

训练完成后，结果将保存在 `results/` 目录：
- `checkpoints/`: 模型检查点
- `logs/`: TensorBoard日志
- `visualizations/`: 可视化结果
- `predictions/`: 预测结果

## 故障排除

### 常见问题

1. **图像加载失败**
   - 数据加载器会自动跳过损坏的图像
   - 检查数据集路径是否正确

2. **内存不足**
   - 减小批次大小
   - 减小图像尺寸
   - 减少工作进程数

3. **训练速度慢**
   - 使用GPU训练
   - 增加批次大小
   - 使用混合精度训练

### 调试模式

设置环境变量启用详细日志：
```bash
export PYTHONPATH=$PWD
python train.py
```

## 贡献

欢迎提交问题和改进建议！

## 许可证

本项目仅供学习和研究使用。