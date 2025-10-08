"""
测试单个样本加载
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_loader import RobustToothDataset
import torch

def test_single_sample():
    """测试单个样本加载"""
    print("=== 测试单个样本加载 ===")
    
    data_dir = "/mnt/external_4tb/hjy/assignment1/data/ToothSegmDataset"
    
    # 创建数据集
    dataset = RobustToothDataset(
        data_dir=data_dir,
        split='train',
        tooth_types=None,
        image_size=256,
        augment=False,
        skip_corrupted=True
    )
    
    print(f"数据集大小: {len(dataset)}")
    
    if len(dataset) > 0:
        # 测试加载前几个样本
        for i in range(min(5, len(dataset))):
            try:
                print(f"\n测试样本 {i}:")
                sample = dataset[i]
                print(f"  样本键: {list(sample.keys())}")
                print(f"  图像形状: {sample['image'].shape}")
                print(f"  图像值范围: {sample['image'].min():.3f} - {sample['image'].max():.3f}")
                if 'mask' in sample:
                    print(f"  掩码形状: {sample['mask'].shape}")
                    print(f"  掩码唯一值: {sample['mask'].unique()}")
                print(f"  牙齿类型: {sample['tooth_type']}")
                print(f"  样本ID: {sample['sample_id']}")
            except Exception as e:
                print(f"  样本 {i} 加载失败: {e}")
    else:
        print("数据集为空！")
    
    # 测试数据加载器
    print("\n=== 测试数据加载器 ===")
    try:
        from data_loader import create_dataloaders
        
        train_loader, val_loader, test_loader = create_dataloaders(
            data_dir=data_dir,
            batch_size=2,
            num_workers=0,  # 使用单进程
            image_size=256,
            tooth_types=None
        )
        
        print(f"训练批次数: {len(train_loader)}")
        print(f"验证批次数: {len(val_loader)}")
        print(f"测试批次数: {len(test_loader)}")
        
        # 测试一个批次
        if len(train_loader) > 0:
            print("\n测试训练批次...")
            for batch in train_loader:
                print(f"批次图像形状: {batch['image'].shape}")
                print(f"批次图像值范围: {batch['image'].min():.3f} - {batch['image'].max():.3f}")
                if 'mask' in batch:
                    print(f"批次掩码形状: {batch['mask'].shape}")
                    print(f"批次掩码唯一值: {batch['mask'].unique()}")
                print(f"牙齿类型: {batch['tooth_type']}")
                break
        else:
            print("训练加载器为空！")
            
    except Exception as e:
        print(f"数据加载器测试失败: {e}")

if __name__ == "__main__":
    test_single_sample()

