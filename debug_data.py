"""
调试数据加载问题
"""
import os
from pathlib import Path
from data_loader import RobustToothDataset

def debug_data_loading():
    """调试数据加载"""
    print("=== 调试数据加载 ===")
    
    data_dir = "/mnt/external_4tb/hjy/assignment1/data/ToothSegmDataset"
    
    # 检查数据目录是否存在
    print(f"数据目录: {data_dir}")
    print(f"数据目录存在: {os.path.exists(data_dir)}")
    
    if os.path.exists(data_dir):
        # 检查子目录
        data_path = Path(data_dir)
        print(f"testset存在: {(data_path / 'testset').exists()}")
        print(f"trainset_valset存在: {(data_path / 'trainset_valset').exists()}")
        
        # 检查testset
        testset_dir = data_path / 'testset'
        if testset_dir.exists():
            test_dirs = [d for d in testset_dir.iterdir() if d.is_dir()]
            print(f"testset子目录数量: {len(test_dirs)}")
            print(f"前5个testset目录: {[d.name for d in test_dirs[:5]]}")
        
        # 检查trainset_valset
        trainset_dir = data_path / 'trainset_valset'
        if trainset_dir.exists():
            train_dirs = [d for d in trainset_dir.iterdir() if d.is_dir()]
            print(f"trainset_valset子目录数量: {len(train_dirs)}")
            print(f"前5个trainset_valset目录: {[d.name for d in train_dirs[:5]]}")
            
            # 检查train和val目录
            if train_dirs:
                sample_dir = train_dirs[0]
                train_path = sample_dir / 'train'
                val_path = sample_dir / 'val'
                print(f"示例目录 {sample_dir.name}:")
                print(f"  train目录存在: {train_path.exists()}")
                print(f"  val目录存在: {val_path.exists()}")
                
                if train_path.exists():
                    train_patients = [d for d in train_path.iterdir() if d.is_dir()]
                    print(f"  train患者数量: {len(train_patients)}")
                    if train_patients:
                        sample_patient = train_patients[0]
                        rgb_files = list(sample_patient.glob('*_rgb.jpg'))
                        mask_files = list(sample_patient.glob('*_mask.jpg'))
                        print(f"  示例患者 {sample_patient.name}:")
                        print(f"    RGB文件数量: {len(rgb_files)}")
                        print(f"    掩码文件数量: {len(mask_files)}")
    
    # 测试数据加载器
    print("\n=== 测试数据加载器 ===")
    
    try:
        # 测试训练集
        print("测试训练集...")
        train_dataset = RobustToothDataset(
            data_dir=data_dir,
            split='train',
            tooth_types=None,  # 自动检测
            image_size=256,
            augment=False,
            skip_corrupted=True
        )
        print(f"训练集样本数: {len(train_dataset)}")
        
        if len(train_dataset) > 0:
            sample = train_dataset[0]
            print(f"样本键: {list(sample.keys())}")
            print(f"图像形状: {sample['image'].shape}")
            if 'mask' in sample:
                print(f"掩码形状: {sample['mask'].shape}")
        
    except Exception as e:
        print(f"训练集加载失败: {e}")
    
    try:
        # 测试验证集
        print("\n测试验证集...")
        val_dataset = RobustToothDataset(
            data_dir=data_dir,
            split='val',
            tooth_types=None,
            image_size=256,
            augment=False,
            skip_corrupted=True
        )
        print(f"验证集样本数: {len(val_dataset)}")
        
    except Exception as e:
        print(f"验证集加载失败: {e}")
    
    try:
        # 测试测试集
        print("\n测试测试集...")
        test_dataset = RobustToothDataset(
            data_dir=data_dir,
            split='test',
            tooth_types=None,
            image_size=256,
            augment=False,
            skip_corrupted=True
        )
        print(f"测试集样本数: {len(test_dataset)}")
        
    except Exception as e:
        print(f"测试集加载失败: {e}")

if __name__ == "__main__":
    debug_data_loading()
