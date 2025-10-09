"""
示例运行脚本 - 快速开始
"""
import os
import torch
import numpy as np
from config import *
from models.unet import create_model
from utils.dataset import create_data_loaders, ToothSegmentationDataset
from utils.visualization import show_anns, visualize_data_samples


def check_environment():
    """检查环境配置"""
    print("=== 环境检查 ===")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA设备数量: {torch.cuda.device_count()}")
        print(f"当前CUDA设备: {torch.cuda.current_device()}")
    
    print(f"数据路径: {DATA_ROOT}")
    print(f"训练数据存在: {os.path.exists(TRAIN_DATA_PATH)}")
    print(f"测试数据存在: {os.path.exists(TEST_DATA_PATH)}")
    print(f"输出目录: {OUTPUT_DIR}")
    print()


def test_data_loading():
    """测试数据加载"""
    print("=== 数据加载测试 ===")
    
    try:
        # 创建小数据集进行测试
        test_dataset = ToothSegmentationDataset(
            data_path=TRAIN_DATA_PATH,
            tooth_ids=[0, 1],  # 只测试前两个牙齿ID
            is_training=True
        )
        
        print(f"数据集大小: {len(test_dataset)}")
        
        if len(test_dataset) > 0:
            # 测试一个样本
            sample = test_dataset[0]
            print(f"图像形状: {sample['image'].shape}")
            print(f"Mask形状: {sample['mask'].shape}")
            print(f"牙齿ID: {sample['tooth_id']}")
            print(f"患者ID: {sample['patient_id']}")
            print("数据加载成功！")
        else:
            print("警告: 数据集为空")
            
    except Exception as e:
        print(f"数据加载失败: {e}")
    
    print()


def test_model():
    """测试模型"""
    print("=== 模型测试 ===")
    
    try:
        # 创建模型
        model = create_model(MODEL_CONFIG)
        print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
        
        # 测试前向传播
        batch_size = 2
        input_image = torch.randn(batch_size, 3, 256, 256)
        tooth_ids = torch.tensor([0, 1])
        
        model.eval()
        with torch.no_grad():
            output = model(input_image, tooth_ids)
        
        print(f"输入形状: {input_image.shape}")
        print(f"输出形状: {output.shape}")
        print("模型测试成功！")
        
    except Exception as e:
        print(f"模型测试失败: {e}")
    
    print()


def test_training_loop():
    """测试训练循环"""
    print("=== 训练循环测试 ===")
    
    try:
        # 创建数据加载器
        train_loader, val_loader = create_data_loaders(
            selected_tooth_ids=[0, 1],
            batch_size=2,
            val_split=0.2
        )
        
        print(f"训练批次数量: {len(train_loader)}")
        print(f"验证批次数量: {len(val_loader)}")
        
        # 测试一个训练批次
        for batch in train_loader:
            print(f"批次图像形状: {batch['image'].shape}")
            print(f"批次Mask形状: {batch['mask'].shape}")
            print(f"批次牙齿ID: {batch['tooth_id']}")
            break
        
        print("训练循环测试成功！")
        
    except Exception as e:
        print(f"训练循环测试失败: {e}")
    
    print()


def test_visualization():
    """测试可视化功能"""
    print("=== 可视化测试 ===")
    
    try:
        # 创建测试数据
        test_dataset = ToothSegmentationDataset(
            data_path=TRAIN_DATA_PATH,
            tooth_ids=[0],
            is_training=False
        )
        
        if len(test_dataset) > 0:
            # 可视化前几个样本
            print("正在生成数据样本可视化...")
            visualize_data_samples(test_dataset, num_samples=min(4, len(test_dataset)))
            print("可视化测试成功！")
        else:
            print("无法进行可视化测试：数据集为空")
            
    except Exception as e:
        print(f"可视化测试失败: {e}")
    
    print()


def main():
    """主函数"""
    print("牙齿分割项目 - 快速开始测试")
    print("=" * 50)
    
    # 检查环境
    check_environment()
    
    # 测试数据加载
    test_data_loading()
    
    # 测试模型
    test_model()
    
    # 测试训练循环
    test_training_loop()
    
    # 测试可视化
    test_visualization()
    
    print("=" * 50)
    print("测试完成！")
    print()
    print("下一步:")
    print("1. 运行训练: python train.py")
    print("2. 运行测试: python test.py --model_path outputs/models/best_model.pth")
    print("3. 查看README.md了解详细使用方法")


if __name__ == "__main__":
    main()
