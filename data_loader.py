"""
健壮的数据加载器，处理损坏的图像文件
"""
import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import cv2
from typing import Tuple, Optional, List, Dict
import logging
import random
from pathlib import Path

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RobustToothDataset(Dataset):
    """
    健壮的牙齿分割数据集，能够处理损坏的图像文件
    """
    
    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        tooth_types: Optional[List[int]] = None,
        image_size: int = 512,
        augment: bool = True,
        skip_corrupted: bool = True
    ):
        """
        初始化数据集
        
        Args:
            data_dir: 数据集根目录
            split: 'train', 'val', 或 'test'
            tooth_types: 要包含的牙齿类型列表
            image_size: 图像尺寸
            augment: 是否进行数据增强
            skip_corrupted: 是否跳过损坏的图像
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.image_size = image_size
        self.augment = augment
        self.skip_corrupted = skip_corrupted
        
        # 默认包含所有牙齿类型
        if tooth_types is None:
            tooth_types = self._get_available_tooth_types()
        
        self.tooth_types = tooth_types
        self.samples = self._load_samples()
        
        logger.info(f"加载了 {len(self.samples)} 个样本 ({split} 集)")
    
    def _get_available_tooth_types(self) -> List[int]:
        """自动检测可用的牙齿类型"""
        tooth_types = []
        
        if self.split == 'test':
            # 检查testset目录
            test_dir = self.data_dir / 'testset'
            if test_dir.exists():
                for item in test_dir.iterdir():
                    if item.is_dir() and item.name.isdigit():
                        tooth_types.append(int(item.name))
        else:
            # 检查trainset_valset目录
            train_val_dir = self.data_dir / 'trainset_valset'
            if train_val_dir.exists():
                for item in train_val_dir.iterdir():
                    if item.is_dir():
                        # 检查是否有对应的split目录
                        split_dir = item / self.split
                        if split_dir.exists():
                            # 检查是否有患者目录和图像文件
                            has_images = False
                            for patient_dir in split_dir.iterdir():
                                if patient_dir.is_dir():
                                    rgb_files = list(patient_dir.glob('*_rgb.jpg'))
                                    if rgb_files:
                                        has_images = True
                                        break
                            if has_images:
                                tooth_types.append(int(item.name))
        
        tooth_types.sort()
        logger.info(f"发现 {len(tooth_types)} 个牙齿类型: {tooth_types[:10]}{'...' if len(tooth_types) > 10 else ''}")
        return tooth_types
    
    def _load_samples(self) -> List[Dict]:
        """加载所有样本路径"""
        samples = []
        
        if self.split == 'test':
            # 测试集结构: testset/0000/, testset/0001/, etc.
            test_dir = self.data_dir / 'testset'
            for tooth_type in self.tooth_types:
                tooth_dir = test_dir / f"{tooth_type:04d}"
                if tooth_dir.exists():
                    for img_file in sorted(tooth_dir.glob('*_rgb.jpg')):
                        samples.append({
                            'tooth_type': tooth_type,
                            'image_path': str(img_file),
                            'mask_path': None,
                            'sample_id': img_file.stem.replace('_rgb', '')
                        })
        else:
            # 训练/验证集结构: trainset_valset/{tooth_type}/{split}/{patient_id}/
            for tooth_type in self.tooth_types:
                tooth_dir = self.data_dir / 'trainset_valset' / str(tooth_type) / self.split
                if tooth_dir.exists():
                    patient_dirs = [d for d in tooth_dir.iterdir() if d.is_dir()]
                    logger.info(f"牙齿类型 {tooth_type}: 找到 {len(patient_dirs)} 个患者目录")
                    
                    for patient_dir in sorted(patient_dirs):
                        if patient_dir.is_dir():
                            # 尝试不同的文件命名模式
                            rgb_files = list(patient_dir.glob('*_rgb.jpg'))
                            if not rgb_files:
                                # 尝试其他可能的命名模式
                                rgb_files = list(patient_dir.glob('*rgb*.jpg'))
                            if not rgb_files:
                                # 尝试所有jpg文件
                                all_jpg = list(patient_dir.glob('*.jpg'))
                                # 过滤掉掩码文件
                                rgb_files = [f for f in all_jpg if 'mask' not in f.name.lower()]
                            
                            logger.info(f"患者 {patient_dir.name}: 找到 {len(rgb_files)} 个RGB文件")
                            
                            for img_file in sorted(rgb_files):
                                # 尝试找到对应的掩码文件
                                mask_file = None
                                if '_rgb' in img_file.name:
                                    mask_file = img_file.parent / img_file.name.replace('_rgb.jpg', '_mask.jpg')
                                else:
                                    # 尝试其他命名模式
                                    base_name = img_file.stem
                                    possible_mask_names = [
                                        base_name + '_mask.jpg',
                                        base_name.replace('rgb', 'mask') + '.jpg',
                                        base_name + 'mask.jpg'
                                    ]
                                    for mask_name in possible_mask_names:
                                        potential_mask = img_file.parent / mask_name
                                        if potential_mask.exists():
                                            mask_file = potential_mask
                                            break
                                
                                samples.append({
                                    'tooth_type': tooth_type,
                                    'image_path': str(img_file),
                                    'mask_path': str(mask_file) if mask_file and mask_file.exists() else None,
                                    'sample_id': img_file.stem,
                                    'patient_id': patient_dir.name
                                })
        
        return samples
    
    def _load_image_safe(self, image_path: str) -> Optional[np.ndarray]:
        """
        安全加载图像，处理损坏的文件
        
        Args:
            image_path: 图像路径
            
        Returns:
            图像数组或None（如果加载失败）
        """
        # 检查文件是否存在
        if not os.path.exists(image_path):
            logger.warning(f"文件不存在: {image_path}")
            return None
        
        # 检查文件大小
        file_size = os.path.getsize(image_path)
        if file_size == 0:
            logger.warning(f"文件为空: {image_path}")
            return None
        
        try:
            # 首先尝试用PIL加载
            with Image.open(image_path) as img:
                img = img.convert('RGB')
                img_array = np.array(img)
                
                # 检查图像是否有效
                if img_array.size == 0 or img_array.shape[0] == 0 or img_array.shape[1] == 0:
                    logger.warning(f"图像尺寸无效: {image_path}")
                    return None
                
                return img_array
        except Exception as e:
            logger.warning(f"PIL加载失败 {image_path}: {e}")
            
            try:
                # 尝试用OpenCV加载
                img = cv2.imread(image_path)
                if img is not None and img.size > 0:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    return img
                else:
                    logger.warning(f"OpenCV加载失败或图像为空: {image_path}")
                    return None
            except Exception as e2:
                logger.warning(f"OpenCV加载异常 {image_path}: {e2}")
                return None
    
    def _load_mask_safe(self, mask_path: str) -> Optional[np.ndarray]:
        """
        安全加载掩码
        
        Args:
            mask_path: 掩码路径
            
        Returns:
            掩码数组或None（如果加载失败）
        """
        if mask_path is None or not os.path.exists(mask_path):
            return None
        
        try:
            with Image.open(mask_path) as mask:
                mask = mask.convert('L')
                return np.array(mask)
        except Exception as e:
            if not self.skip_corrupted:
                logger.warning(f"加载掩码失败 {mask_path}: {e}")
            return None
    
    def _resize_image(self, image: np.ndarray, target_size: int) -> np.ndarray:
        """调整图像大小"""
        return cv2.resize(image, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    
    def _resize_mask(self, mask: np.ndarray, target_size: int) -> np.ndarray:
        """调整掩码大小"""
        return cv2.resize(mask, (target_size, target_size), interpolation=cv2.INTER_NEAREST)
    
    def _augment_image(self, image: np.ndarray, mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """数据增强"""
        if not self.augment:
            return image, mask
        
        # 随机水平翻转
        if random.random() > 0.5:
            image = cv2.flip(image, 1)
            if mask is not None:
                mask = cv2.flip(mask, 1)
        
        # 随机垂直翻转
        if random.random() > 0.5:
            image = cv2.flip(image, 0)
            if mask is not None:
                mask = cv2.flip(mask, 0)
        
        # 随机旋转90度
        if random.random() > 0.5:
            k = random.randint(1, 3)
            image = np.rot90(image, k)
            if mask is not None:
                mask = np.rot90(mask, k)
        
        # 随机亮度调整
        if random.random() > 0.5:
            alpha = random.uniform(0.8, 1.2)
            image = np.clip(image * alpha, 0, 255).astype(np.uint8)
        
        return image, mask
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        
        # 加载图像
        image = self._load_image_safe(sample['image_path'])
        if image is None:
            # 如果图像加载失败且允许跳过，尝试下一个样本（最多尝试10次）
            if self.skip_corrupted:
                for attempt in range(10):
                    next_idx = (idx + attempt + 1) % len(self.samples)
                    next_sample = self.samples[next_idx]
                    next_image = self._load_image_safe(next_sample['image_path'])
                    if next_image is not None:
                        sample = next_sample
                        image = next_image
                        break
                else:
                    # 如果10次尝试都失败，创建占位图像
                    logger.warning(f"连续10次尝试加载图像都失败，使用占位图像")
                    image = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
            else:
                # 创建占位图像
                image = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        
        # 加载掩码
        mask = None
        if sample['mask_path']:
            mask = self._load_mask_safe(sample['mask_path'])
            if mask is None and not self.skip_corrupted:
                # 创建占位掩码
                mask = np.zeros((self.image_size, self.image_size), dtype=np.uint8)
        
        # 调整大小
        image = self._resize_image(image, self.image_size)
        if mask is not None:
            mask = self._resize_mask(mask, self.image_size)
        
        # 数据增强
        image, mask = self._augment_image(image, mask)
        
        # 转换为张量
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        
        result = {
            'image': image,
            'tooth_type': sample['tooth_type'],
            'sample_id': sample['sample_id']
        }
        
        if mask is not None:
            # 二值化掩码
            mask = (mask > 128).astype(np.uint8)
            result['mask'] = torch.from_numpy(mask).long()
        
        return result


def create_dataloaders(
    data_dir: str,
    batch_size: int = 8,
    num_workers: int = 4,
    image_size: int = 512,
    tooth_types: Optional[List[int]] = None,
    train_augment: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    创建数据加载器
    
    Args:
        data_dir: 数据集目录
        batch_size: 批次大小
        num_workers: 工作进程数
        image_size: 图像尺寸
        tooth_types: 牙齿类型列表
        train_augment: 训练时是否增强
    
    Returns:
        (train_loader, val_loader, test_loader)
    """
    # 创建数据集
    train_dataset = RobustToothDataset(
        data_dir=data_dir,
        split='train',
        tooth_types=tooth_types,
        image_size=image_size,
        augment=train_augment,
        skip_corrupted=True
    )
    
    val_dataset = RobustToothDataset(
        data_dir=data_dir,
        split='val',
        tooth_types=tooth_types,
        image_size=image_size,
        augment=False,
        skip_corrupted=True
    )
    
    test_dataset = RobustToothDataset(
        data_dir=data_dir,
        split='test',
        tooth_types=tooth_types,
        image_size=image_size,
        augment=False,
        skip_corrupted=True
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


def test_dataset():
    """测试数据集"""
    print("=== 测试数据集 ===")
    
    data_dir = "/Users/oguya/Documents/Courses/6004-image/assignment1/data/ToothSegmDataset"
    
    # 测试训练集
    print("测试训练集...")
    train_dataset = RobustToothDataset(
        data_dir=data_dir,
        split='train',
        tooth_types=[0, 1, 2],  # 只测试前3个牙齿类型
        image_size=512,
        augment=True,
        skip_corrupted=True
    )
    
    print(f"训练集样本数: {len(train_dataset)}")
    
    if len(train_dataset) > 0:
        sample = train_dataset[0]
        print(f"样本键: {list(sample.keys())}")
        print(f"图像形状: {sample['image'].shape}")
        print(f"图像值范围: {sample['image'].min():.3f} - {sample['image'].max():.3f}")
        if 'mask' in sample:
            print(f"掩码形状: {sample['mask'].shape}")
            print(f"掩码唯一值: {torch.unique(sample['mask'])}")
    
    # 测试测试集
    print("\n测试测试集...")
    test_dataset = RobustToothDataset(
        data_dir=data_dir,
        split='test',
        tooth_types=[0, 1, 2],
        image_size=512,
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
        image_size=512,
        tooth_types=[0, 1, 2]
    )
    
    print(f"训练批次数: {len(train_loader)}")
    print(f"验证批次数: {len(val_loader)}")
    print(f"测试批次数: {len(test_loader)}")
    
    # 测试一个批次
    for batch in train_loader:
        print(f"批次图像形状: {batch['image'].shape}")
        if 'mask' in batch:
            print(f"批次掩码形状: {batch['mask'].shape}")
        break
    
    print("✅ 数据集测试完成！")


if __name__ == "__main__":
    test_dataset()
