"""
牙齿分割数据集类
"""
import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from config import TRAIN_DATA_PATH, TEST_DATA_PATH, TOOTH_ID_MAPPING, TRAIN_CONFIG, AUGMENTATION_CONFIG


def read_mask(mask_path):
    """
    读取mask图像并分离不同区域
    Args:
        mask_path: mask图像路径
    Returns:
        mask_upper: 上牙龈mask (value 1 -> 255)
        mask_lower: 下牙龈mask (value 2 -> 255) 
        mask_tooth: 目标牙齿mask (value 3 -> 255)
    """
    mask_img = cv2.imread(mask_path, -1)
    if mask_img is None:
        raise ValueError(f"无法读取mask图像: {mask_path}")
    
    # 分离不同区域
    mask_upper = np.where(mask_img == 1, 255, 0).astype(np.uint8)
    mask_lower = np.where(mask_img == 2, 255, 0).astype(np.uint8)
    mask_tooth = np.where(mask_img == 3, 255, 0).astype(np.uint8)
    
    return mask_upper, mask_lower, mask_tooth


class ToothSegmentationDataset(Dataset):
    """
    牙齿分割数据集类
    """
    def __init__(self, data_path, tooth_ids, is_training=True, image_size=(256, 256)):
        """
        Args:
            data_path: 数据路径
            tooth_ids: 要使用的牙齿ID列表
            is_training: 是否为训练模式
            image_size: 图像尺寸
        """
        self.data_path = data_path
        self.tooth_ids = tooth_ids
        self.is_training = is_training
        self.image_size = image_size
        
        # 收集所有数据样本
        self.samples = self._collect_samples()
        
        # 数据增强
        if is_training:
            self.transform = A.Compose([
                A.Resize(image_size[0], image_size[1], always_apply=True),  # 强制调整尺寸
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Rotate(limit=AUGMENTATION_CONFIG['rotation_range'], p=0.5),
                A.RandomBrightnessContrast(
                    brightness_limit=AUGMENTATION_CONFIG['brightness_range'],
                    contrast_limit=AUGMENTATION_CONFIG['contrast_range'],
                    p=0.5
                ),
                A.RandomScale(scale_limit=AUGMENTATION_CONFIG['scale_range'], p=0.5),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
        else:
            self.transform = A.Compose([
                A.Resize(image_size[0], image_size[1], always_apply=True),  # 强制调整尺寸
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
    
    def _collect_samples(self):
        """收集所有数据样本"""
        samples = []
        
        for tooth_id in self.tooth_ids:
            tooth_dir = os.path.join(self.data_path, str(tooth_id))
            if not os.path.exists(tooth_dir):
                print(f"警告: 牙齿ID {tooth_id} 的目录不存在: {tooth_dir}")
                continue
                
            # 查找train目录
            train_dir = os.path.join(tooth_dir, "train")
            if not os.path.exists(train_dir):
                print(f"警告: 牙齿ID {tooth_id} 的train目录不存在: {train_dir}")
                continue
            
            # 遍历所有患者目录
            for patient_dir in os.listdir(train_dir):
                patient_path = os.path.join(train_dir, patient_dir)
                if not os.path.isdir(patient_path):
                    continue
                
                # 查找RGB和mask图像对
                for file in os.listdir(patient_path):
                    if file.endswith('_rgb.jpg'):
                        rgb_path = os.path.join(patient_path, file)
                        mask_file = file.replace('_rgb.jpg', '_mask.jpg')
                        mask_path = os.path.join(patient_path, mask_file)
                        
                        if os.path.exists(mask_path):
                            samples.append({
                                'rgb_path': rgb_path,
                                'mask_path': mask_path,
                                'tooth_id': tooth_id,
                                'patient_id': patient_dir,
                                'image_id': file.replace('_rgb.jpg', '')
                            })
        
        print(f"收集到 {len(samples)} 个训练样本")
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 读取RGB图像
        rgb_image = cv2.imread(sample['rgb_path'])
        rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
        
        # 读取mask
        mask_upper, mask_lower, mask_tooth = read_mask(sample['mask_path'])
        
        # 创建目标牙齿的二进制mask
        target_mask = mask_tooth
        
        # 应用数据增强
        transformed = self.transform(image=rgb_image, mask=target_mask)
        
        rgb_tensor = transformed['image']
        mask_tensor = transformed['mask'].float() / 255.0  # 归一化到[0,1]
        mask_tensor = mask_tensor.unsqueeze(0)  # 添加通道维度
        
        # 牙齿ID转换为tensor
        tooth_id_tensor = torch.tensor(sample['tooth_id'], dtype=torch.long)
        
        return {
            'image': rgb_tensor,
            'mask': mask_tensor,
            'tooth_id': tooth_id_tensor,
            'patient_id': sample['patient_id'],
            'image_id': sample['image_id']
        }


class TestDataset(Dataset):
    """
    测试数据集类
    """
    def __init__(self, data_path, image_size=(256, 256)):
        self.data_path = data_path
        self.image_size = image_size
        
        # 收集测试样本
        self.samples = self._collect_test_samples()
        
        # 测试时的变换
        self.transform = A.Compose([
            A.Resize(image_size[0], image_size[1]),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    
    def _collect_test_samples(self):
        """收集测试样本"""
        samples = []
        
        for patient_dir in os.listdir(self.data_path):
            patient_path = os.path.join(self.data_path, patient_dir)
            if not os.path.isdir(patient_path):
                continue
            
            for file in os.listdir(patient_path):
                if file.endswith('_rgb.jpg'):
                    rgb_path = os.path.join(patient_path, file)
                    samples.append({
                        'rgb_path': rgb_path,
                        'patient_id': patient_dir,
                        'image_id': file.replace('_rgb.jpg', '')
                    })
        
        print(f"收集到 {len(samples)} 个测试样本")
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 读取RGB图像
        rgb_image = cv2.imread(sample['rgb_path'])
        rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
        
        # 应用变换
        transformed = self.transform(image=rgb_image)
        rgb_tensor = transformed['image']
        
        return {
            'image': rgb_tensor,
            'patient_id': sample['patient_id'],
            'image_id': sample['image_id']
        }


def create_data_loaders(selected_tooth_ids, batch_size=8, val_split=0.1, num_workers=4):
    """
    创建训练和验证数据加载器
    """
    # 创建完整数据集
    full_dataset = ToothSegmentationDataset(
        data_path=TRAIN_DATA_PATH,
        tooth_ids=selected_tooth_ids,
        is_training=True
    )
    
    # 计算训练和验证集大小
    total_size = len(full_dataset)
    val_size = int(total_size * val_split)
    train_size = total_size - val_size
    
    # 随机分割数据集
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # 设置为0避免多进程问题
        pin_memory=False  # 在MPS上禁用pin_memory
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,  # 设置为0避免多进程问题
        pin_memory=False  # 在MPS上禁用pin_memory
    )
    
    return train_loader, val_loader


def create_test_loader(batch_size=8, num_workers=4):
    """
    创建测试数据加载器
    """
    test_dataset = TestDataset(data_path=TEST_DATA_PATH)
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,  # 设置为0避免多进程问题
        pin_memory=False  # 在MPS上禁用pin_memory
    )
    
    return test_loader
