"""
DataLoader模块 - 数据加载器类
统一使用类式设计，提供更好的封装和状态管理
"""

import os
import sys
import torch
from torch.utils.data import DataLoader

# 添加src目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataset import ToothSegmentationDataset
from transforms import get_train_transforms, get_val_transforms, get_mask_transforms

class ToothDataLoader:
    """
    牙齿分割数据加载器类
    
    功能：
    - 管理训练、验证、测试数据加载器
    - 提供统一的数据加载接口
    - 支持自定义配置和参数
    """
    
    def __init__(self, data_dir='../data/ToothSegmDataset', batch_size=8, num_workers=4):
        """
        初始化数据加载器
        
        Args:
            data_dir (str): 数据集根目录路径
            batch_size (int): 批次大小
            num_workers (int): 数据加载进程数
        """
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # 数据加载器
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        
        # 数据集
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
        print(f"DataLoader初始化完成")
        print(f"  数据目录: {self.data_dir}")
        print(f"  批次大小: {self.batch_size}")
        print(f"  工作进程: {self.num_workers}")
    
    def create_datasets(self, use_transforms=True):
        """
        创建数据集
        
        Args:
            use_transforms (bool): 是否使用数据变换
        """
        print("创建数据集...")
        
        if use_transforms:
            # 使用数据变换
            train_transform = get_train_transforms()
            val_transform = get_val_transforms()
            mask_transform = get_mask_transforms()
        else:
            # 不使用变换
            train_transform = val_transform = mask_transform = None
        
        self.train_dataset = ToothSegmentationDataset(
            self.data_dir, mode='train', 
            transform=train_transform, mask_transform=mask_transform
        )
        
        self.val_dataset = ToothSegmentationDataset(
            self.data_dir, mode='val', 
            transform=val_transform, mask_transform=mask_transform
        )
        
        self.test_dataset = ToothSegmentationDataset(
            self.data_dir, mode='test', 
            transform=val_transform, mask_transform=None
        )
        
        print(f"  训练集: {len(self.train_dataset)} 样本")
        print(f"  验证集: {len(self.val_dataset)} 样本")
        print(f"  测试集: {len(self.test_dataset)} 样本")
    
    def create_loaders(self, shuffle_train=True, shuffle_val=False, shuffle_test=False,
                      drop_last_train=True, drop_last_val=False, drop_last_test=False):
        """
        创建数据加载器
        
        Args:
            shuffle_train (bool): 训练集是否打乱
            shuffle_val (bool): 验证集是否打乱
            shuffle_test (bool): 测试集是否打乱
            drop_last_train (bool): 训练集是否丢弃最后批次
            drop_last_val (bool): 验证集是否丢弃最后批次
            drop_last_test (bool): 测试集是否丢弃最后批次
        """
        if self.train_dataset is None:
            raise ValueError("请先调用 create_datasets() 创建数据集")
        
        print("创建数据加载器...")
        
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=shuffle_train,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=drop_last_train
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=shuffle_val,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=drop_last_val
        )
        
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=shuffle_test,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=drop_last_test
        )
        
        print(f"  训练加载器: {len(self.train_loader)} 批次")
        print(f"  验证加载器: {len(self.val_loader)} 批次")
        print(f"  测试加载器: {len(self.test_loader)} 批次")
    
    def get_train_loader(self):
        """获取训练数据加载器"""
        if self.train_loader is None:
            raise ValueError("请先调用 create_loaders() 创建数据加载器")
        return self.train_loader
    
    def get_val_loader(self):
        """获取验证数据加载器"""
        if self.val_loader is None:
            raise ValueError("请先调用 create_loaders() 创建数据加载器")
        return self.val_loader
    
    def get_test_loader(self):
        """获取测试数据加载器"""
        if self.test_loader is None:
            raise ValueError("请先调用 create_loaders() 创建数据加载器")
        return self.test_loader
    
    def get_all_loaders(self):
        """获取所有数据加载器"""
        return self.get_train_loader(), self.get_val_loader(), self.get_test_loader()
    
    def get_dataset_info(self):
        """获取数据集信息"""
        info = {
            'train_samples': len(self.train_dataset) if self.train_dataset else 0,
            'val_samples': len(self.val_dataset) if self.val_dataset else 0,
            'test_samples': len(self.test_dataset) if self.test_dataset else 0,
            'batch_size': self.batch_size,
            'num_workers': self.num_workers
        }
        return info
    
    def update_batch_size(self, new_batch_size):
        """更新批次大小并重新创建加载器"""
        self.batch_size = new_batch_size
        if self.train_dataset is not None:
            self.create_loaders()
            print(f"批次大小已更新为: {self.batch_size}")
    
    def test_loading(self, num_batches=3):
        """
        测试数据加载
        
        Args:
            num_batches (int): 测试的批次数量
        """
        print(f"测试数据加载 (前{num_batches}个批次)...")
        
        try:
            train_loader = self.get_train_loader()
            
            for i, batch in enumerate(train_loader):
                print(f"批次 {i+1}:")
                print(f"  图像数量: {len(batch['images'])}")
                print(f"  掩码数量: {len(batch['masks']) if 'masks' in batch else 0}")
                print(f"  样本ID: {batch['sample_ids'][:3]}...")
                
                if i >= num_batches - 1:
                    break
            
            print("✅ 数据加载测试成功！")
            
        except Exception as e:
            print(f"❌ 数据加载测试失败: {e}")
            import traceback
            traceback.print_exc()

def test_data_loader_class():
    """测试DataLoader类"""
    print("=== 测试DataLoader类 ===")
    
    try:
        # 创建DataLoader实例
        data_loader = ToothDataLoader(
            data_dir='../data/ToothSegmDataset',
            batch_size=4,
            num_workers=2
        )
        
        # 创建数据集（使用变换）
        data_loader.create_datasets(use_transforms=True)
        
        # 创建数据加载器
        data_loader.create_loaders()
        
        # 获取数据集信息
        info = data_loader.get_dataset_info()
        print(f"\n数据集信息: {info}")
        
        # 测试数据加载
        data_loader.test_loading(num_batches=2)
        
        # 测试更新批次大小
        data_loader.update_batch_size(8)
        
        print("\n✅ DataLoader类测试完成！")
        
    except Exception as e:
        print(f"❌ DataLoader类测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_data_loader_class()
