"""
测试数据加载器
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_loader import RobustToothDataset, create_dataloaders
import matplotlib.pyplot as plt
import numpy as np


def test_data_loading():
    """测试数据加载"""
    print("=== 测试数据加载 ===")
    
    data_dir = "/Users/oguya/Documents/Courses/6004-image/assignment1/data/ToothSegmDataset"
    
    # 测试训练集
    print("测试训练集...")
    train_dataset = RobustToothDataset(
        data_dir=data_dir,
        split='train',
        tooth_types=[0, 1, 2],  # 只测试前3个牙齿类型
        image_size=256,
        augment=True,
        skip_corrupted=True
    )
    
    print(f"训练集样本数: {len(train_dataset)}")
    
    if len(train_dataset) > 0:
        # 测试加载一个样本
        sample = train_dataset[0]
        print(f"样本键: {list(sample.keys())}")
        print(f"图像形状: {sample['image'].shape}")
        print(f"图像值范围: {sample['image'].min():.3f} - {sample['image'].max():.3f}")
        if 'mask' in sample:
            print(f"掩码形状: {sample['mask'].shape}")
            print(f"掩码唯一值: {sample['mask'].unique()}")
        
        # 可视化几个样本
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        for i in range(min(6, len(train_dataset))):
            sample = train_dataset[i]
            row = i // 3
            col = i % 3
            
            # 显示图像
            image = sample['image'].permute(1, 2, 0).numpy()
            axes[row, col].imshow(image)
            axes[row, col].set_title(f"Sample {i} - Tooth Type: {sample['tooth_type']}")
            axes[row, col].axis('off')
        
        plt.tight_layout()
        plt.savefig('test_samples.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("样本可视化已保存为 test_samples.png")
    
    # 测试测试集
    print("\n测试测试集...")
    test_dataset = RobustToothDataset(
        data_dir=data_dir,
        split='test',
        tooth_types=[0, 1, 2],
        image_size=256,
        augment=False,
        skip_corrupted=True
    )
    
    print(f"测试集样本数: {len(test_dataset)}")
    
    if len(test_dataset) > 0:
        sample = test_dataset[0]
        print(f"样本键: {list(sample.keys())}")
        print(f"图像形状: {sample['image'].shape}")
    
    # 测试数据加载器
    print("\n测试数据加载器...")
    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir=data_dir,
        batch_size=4,
        num_workers=2,
        image_size=256,
        tooth_types=[0, 1, 2]
    )
    
    print(f"训练批次数: {len(train_loader)}")
    print(f"验证批次数: {len(val_loader)}")
    print(f"测试批次数: {len(test_loader)}")
    
    # 测试一个批次
    print("\n测试批次加载...")
    for batch in train_loader:
        print(f"批次图像形状: {batch['image'].shape}")
        print(f"批次图像值范围: {batch['image'].min():.3f} - {batch['image'].max():.3f}")
        if 'mask' in batch:
            print(f"批次掩码形状: {batch['mask'].shape}")
            print(f"批次掩码唯一值: {batch['mask'].unique()}")
        print(f"牙齿类型: {batch['tooth_type']}")
        break
    
    print("✅ 数据加载测试完成！")


def test_corrupted_image_handling():
    """测试损坏图像处理"""
    print("\n=== 测试损坏图像处理 ===")
    
    data_dir = "/Users/oguya/Documents/Courses/6004-image/assignment1/data/ToothSegmDataset"
    
    # 创建数据集，不跳过损坏图像
    dataset = RobustToothDataset(
        data_dir=data_dir,
        split='train',
        tooth_types=[30, 29, 21, 5],  # 包含有问题的牙齿类型
        image_size=256,
        augment=False,
        skip_corrupted=False  # 不跳过损坏图像
    )
    
    print(f"数据集样本数: {len(dataset)}")
    
    # 测试加载样本
    success_count = 0
    error_count = 0
    
    for i in range(min(10, len(dataset))):
        try:
            sample = dataset[i]
            if sample['image'].sum() > 0:  # 检查是否成功加载
                success_count += 1
            else:
                error_count += 1
        except Exception as e:
            error_count += 1
            print(f"样本 {i} 加载失败: {e}")
    
    print(f"成功加载: {success_count}")
    print(f"加载失败: {error_count}")
    
    print("✅ 损坏图像处理测试完成！")


if __name__ == "__main__":
    test_data_loading()
    test_corrupted_image_handling()
