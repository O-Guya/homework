"""
配置文件 - 牙齿分割项目
"""
import os

# 数据路径配置
DATA_ROOT = "/Users/oguya/Documents/Courses/6004-image/homework/data/ToothSegmDataset"
TRAIN_DATA_PATH = os.path.join(DATA_ROOT, "trainset_valset")
TEST_DATA_PATH = os.path.join(DATA_ROOT, "testset")

# 牙齿ID映射 (根据prompt中的要求)
TOOTH_ID_MAPPING = {
    "11": 0, "12": 1, "13": 2, "14": 3, "15": 4, "16": 5, "17": 6,
    "21": 8, "22": 9, "23": 10, "24": 11, "25": 12, "26": 13, "27": 14,
    "34": 19, "35": 20, "36": 21, "37": 22,
    "45": 28, "46": 29, "47": 30
}

# 反向映射
ID_TO_TOOTH_MAPPING = {v: k for k, v in TOOTH_ID_MAPPING.items()}

# 模型配置
MODEL_CONFIG = {
    "input_channels": 3,  # RGB图像
    "num_classes": 1,     # 二分类分割任务
    "base_filters": 64,   # 基础滤波器数量
    "dropout_rate": 0.1,  # Dropout率
    "num_tooth_ids": 21,  # 牙齿ID数量
    "tooth_id_embedding_dim": 32,  # 牙齿ID嵌入维度
}

# 训练配置
TRAIN_CONFIG = {
    "batch_size": 8,
    "num_epochs": 100,
    "learning_rate": 1e-4,
    "weight_decay": 1e-5,
    "patience": 15,  # 早停耐心值
    "val_split": 0.1,  # 验证集比例 (10 samples per tooth ID)
    "image_size": (256, 256),  # 输入图像尺寸
    "selected_tooth_ids": [0, 1, 2, 3],  # 选择至少4个牙齿ID进行训练
}

# 数据增强配置
AUGMENTATION_CONFIG = {
    "horizontal_flip": True,
    "vertical_flip": True,
    "rotation_range": 15,
    "brightness_range": 0.2,
    "contrast_range": 0.2,
    "scale_range": (0.8, 1.2),
}

# 输出路径配置
OUTPUT_DIR = "/Users/oguya/Documents/Courses/6004-image/homework/outputs"
MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, "models")
LOG_SAVE_PATH = os.path.join(OUTPUT_DIR, "logs")
RESULT_SAVE_PATH = os.path.join(OUTPUT_DIR, "results")
VISUALIZATION_SAVE_PATH = os.path.join(OUTPUT_DIR, "visualizations")

# 创建输出目录
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
os.makedirs(LOG_SAVE_PATH, exist_ok=True)
os.makedirs(RESULT_SAVE_PATH, exist_ok=True)
os.makedirs(VISUALIZATION_SAVE_PATH, exist_ok=True)