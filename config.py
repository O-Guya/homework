"""
配置文件
"""
import os
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).parent

# 数据路径
DATA_DIR = PROJECT_ROOT / "data" / "ToothSegmDataset"
OUTPUT_DIR = PROJECT_ROOT / "results"

# 模型配置
MODEL_CONFIG = {
    'num_classes': 2,  # 背景 + 牙齿
    'image_size': 512,
    'batch_size': 8,
    'num_workers': 4,
    'epochs': 100,
    'learning_rate': 1e-4,
    'weight_decay': 1e-5,
    'patience': 10,
    'early_stopping_patience': 20,
    'tooth_types': None,  # 使用所有牙齿类型
}

# 数据增强配置
AUGMENTATION_CONFIG = {
    'horizontal_flip': True,
    'vertical_flip': True,
    'rotation': True,
    'brightness_contrast': True,
    'noise': False,
    'blur': False,
}

# 训练配置
TRAINING_CONFIG = {
    'use_mixed_precision': True,
    'gradient_clip_norm': 1.0,
    'save_every_n_epochs': 10,
    'validate_every_n_epochs': 1,
    'log_every_n_steps': 50,
}

# 设备配置
DEVICE_CONFIG = {
    'device': 'cuda' if os.environ.get('CUDA_VISIBLE_DEVICES') else 'cpu',
    'mixed_precision': True,
}

# 输出配置
OUTPUT_CONFIG = {
    'save_predictions': True,
    'save_visualizations': True,
    'save_model_architecture': True,
    'tensorboard_logging': True,
}

# 验证配置
VALIDATION_CONFIG = {
    'metrics': ['dice', 'iou', 'pixel_accuracy', 'precision', 'recall', 'f1'],
    'save_best_model': True,
    'save_last_model': True,
}

# 测试配置
TEST_CONFIG = {
    'batch_size': 16,
    'save_predictions': True,
    'num_visualization_samples': 10,
}

# 日志配置
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': OUTPUT_DIR / 'training.log',
}

# 创建输出目录
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / 'logs').mkdir(exist_ok=True)
(OUTPUT_DIR / 'checkpoints').mkdir(exist_ok=True)
(OUTPUT_DIR / 'visualizations').mkdir(exist_ok=True)
(OUTPUT_DIR / 'predictions').mkdir(exist_ok=True)
