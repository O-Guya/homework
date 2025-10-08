import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class ToothSegmentationDataset(Dataset):
    """
    牙齿分割数据集类
    
    这个类继承自PyTorch的Dataset，用于加载和处理牙齿分割数据
    主要功能：
    1. 加载RGB图像和对应的掩码
    2. 应用数据变换（如resize、normalize等）
    3. 支持训练和测试模式
    """
    
    def __init__(self, data_dir, mode='train', transform=None, mask_transform=None):
        """
        初始化数据集
        
        Args:
            data_dir (str): 数据集根目录路径
            mode (str): 数据集模式 ('train', 'val', 'test')
            transform: RGB图像的变换
            mask_transform: 掩码图像的变换
        """
        self.data_dir = data_dir
        self.mode = mode
        self.transform = transform
        self.mask_transform = mask_transform
        
        # 存储所有样本的路径
        self.samples = []
        
        # 根据模式加载数据
        self._load_samples()
    
    def _load_samples(self):
        """
        加载所有样本的路径信息
        这个函数会遍历数据集目录，找到所有的RGB图像文件
        """
        if self.mode == 'test':
            # 测试集：只有RGB图像，没有掩码
            test_dir = os.path.join(self.data_dir, 'testset')
            for sample_id in sorted(os.listdir(test_dir)):
                sample_path = os.path.join(test_dir, sample_id)
                if os.path.isdir(sample_path):
                    # 获取该样本的所有RGB图像
                    rgb_files = [f for f in os.listdir(sample_path) if f.endswith('_rgb.jpg')]
                    for rgb_file in sorted(rgb_files):
                        rgb_path = os.path.join(sample_path, rgb_file)
                        self.samples.append({
                            'rgb_path': rgb_path,
                            'mask_path': None,  # 测试集没有掩码
                            'sample_id': sample_id,
                            'image_id': rgb_file.replace('_rgb.jpg', '')
                        })
        else:
            # 训练集和验证集：有RGB图像和掩码
            mode_dir = os.path.join(self.data_dir, 'trainset_valset')
            
            # 遍历所有类别
            for class_id in sorted(os.listdir(mode_dir)):
                class_path = os.path.join(mode_dir, class_id)
                if not os.path.isdir(class_path):
                    continue
                    
                # 根据模式选择子目录
                subdir = 'train' if self.mode == 'train' else 'val'
                subdir_path = os.path.join(class_path, subdir)
                
                if not os.path.exists(subdir_path):
                    continue
                
                # 遍历该类别下的所有样本
                for sample_id in os.listdir(subdir_path):
                    sample_path = os.path.join(subdir_path, sample_id)
                    if not os.path.isdir(sample_path):
                        continue
                    
                    # 获取该样本的所有RGB图像
                    rgb_files = [f for f in os.listdir(sample_path) if f.endswith('_rgb.jpg')]
                    for rgb_file in sorted(rgb_files):
                        rgb_path = os.path.join(sample_path, rgb_file)
                        mask_file = rgb_file.replace('_rgb.jpg', '_mask.jpg')
                        mask_path = os.path.join(sample_path, mask_file)
                        
                        # 确保掩码文件存在
                        if os.path.exists(mask_path):
                            self.samples.append({
                                'rgb_path': rgb_path,
                                'mask_path': mask_path,
                                'sample_id': sample_id,
                                'class_id': class_id,
                                'image_id': rgb_file.replace('_rgb.jpg', '')
                            })
        
        print(f"加载了 {len(self.samples)} 个{self.mode}样本")
    
    def __len__(self):
        """返回数据集大小"""
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        获取指定索引的数据样本
        
        Args:
            idx (int): 样本索引
            
        Returns:
            dict: 包含RGB图像、掩码（如果有）和元数据的字典
        """
        sample = self.samples[idx]
        
        # 加载RGB图像
        rgb_image = self._load_image(sample['rgb_path'])
        
        # 加载掩码（如果有）
        mask = None
        if sample['mask_path'] is not None:
            mask = self._load_mask(sample['mask_path'])
        
        # 应用变换
        if self.transform:
            rgb_image = self.transform(rgb_image)
        
        if self.mask_transform and mask is not None:
            mask = self.mask_transform(mask)
        
        # 返回数据字典
        result = {
            'image': rgb_image,
            'sample_id': sample['sample_id'],
            'image_id': sample['image_id']
        }
        
        if mask is not None:
            result['mask'] = mask
            result['class_id'] = sample['class_id']
        
        return result
    
    def _load_image(self, image_path):
        """
        加载RGB图像
        
        Args:
            image_path (str): 图像文件路径
            
        Returns:
            PIL.Image: RGB图像
        """
        try:
            # 使用PIL加载图像，确保是RGB格式
            image = Image.open(image_path).convert('RGB')
            return image
        except Exception as e:
            print(f"加载图像失败 {image_path}: {e}")
            # 返回一个空白图像作为fallback
            return Image.new('RGB', (512, 512), (0, 0, 0))
    
    def _load_mask(self, mask_path):
        """
        加载掩码图像
        
        Args:
            mask_path (str): 掩码文件路径
            
        Returns:
            PIL.Image: 灰度掩码图像
        """
        try:
            # 使用PIL加载掩码，确保是灰度格式
            mask = Image.open(mask_path).convert('L')
            return mask
        except Exception as e:
            print(f"加载掩码失败 {mask_path}: {e}")
            # 返回一个空白掩码作为fallback
            return Image.new('L', (512, 512), 0)

# 测试函数
def test_dataset():
    """测试数据集类的基本功能"""
    print("=== 测试数据集类 ===")
    
    # 创建数据集实例
    dataset = ToothSegmentationDataset(
        data_dir='ToothSegmDataset',
        mode='train'
    )
    
    print(f"数据集大小: {len(dataset)}")
    
    # 测试获取一个样本
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"样本键: {sample.keys()}")
        print(f"图像类型: {type(sample['image'])}")
        print(f"图像尺寸: {sample['image'].size}")
        if 'mask' in sample:
            print(f"掩码类型: {type(sample['mask'])}")
            print(f"掩码尺寸: {sample['mask'].size}")

if __name__ == "__main__":
    test_dataset()
