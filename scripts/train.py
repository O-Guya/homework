"""
训练脚本
最简版本，只包含核心功能
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import sys
import time
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import json
from datetime import datetime

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.unet import UNet
from models.loss import CombinedLoss
from models.optimizer import create_optimizer, create_scheduler
from utils.dataloader import ToothDataLoader

class Trainer:
    """
    训练器类
    管理模型训练过程
    """
    
    def __init__(self, model, criterion, optimizer, scheduler, device):
        """
        初始化训练器
        
        Args:
            model: 模型
            criterion: 损失函数
            optimizer: 优化器
            scheduler: 学习率调度器
            device: 设备
        """
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        
        # 训练历史
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []
        self.epoch_times = []
        self.best_val_loss = float('inf')
        
        # 实验记录
        self.experiment_log = {
            'model_name': 'UNet',
            'num_classes': 5,
            'start_time': datetime.now().isoformat(),
            'device': str(device),
            'hyperparameters': {}
        }
    
    def _pil_to_tensor(self, images, masks):
        """
        将PIL图像或张量转换为张量
        
        Args:
            images: PIL图像列表或张量列表
            masks: PIL掩码列表或张量列表
            
        Returns:
            tuple: (图像张量, 掩码张量)
        """
        # 如果已经是张量，直接返回
        if isinstance(images, torch.Tensor) and isinstance(masks, torch.Tensor):
            return images, masks
        
        # 图像变换
        image_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # 掩码变换
        mask_transform = transforms.Compose([
            transforms.ToTensor()
        ])
        
        # 转换图像
        image_tensors = []
        for img in images:
            if isinstance(img, Image.Image):
                img_tensor = image_transform(img)
            elif isinstance(img, torch.Tensor):
                img_tensor = img
            else:  # 已经是numpy数组
                img_tensor = torch.from_numpy(img).float()
                if img_tensor.dim() == 3:
                    img_tensor = img_tensor.permute(2, 0, 1)  # HWC -> CHW
            image_tensors.append(img_tensor)
        
        # 转换掩码
        mask_tensors = []
        for mask in masks:
            if isinstance(mask, Image.Image):
                mask_tensor = mask_transform(mask).squeeze(0)  # 移除通道维度
            elif isinstance(mask, torch.Tensor):
                mask_tensor = mask
            else:  # 已经是numpy数组
                mask_tensor = torch.from_numpy(mask).float()
            mask_tensors.append(mask_tensor)
        
        # 堆叠成批次
        images = torch.stack(image_tensors)
        masks = torch.stack(mask_tensors)
        
        return images, masks
    
    def train_epoch(self, train_loader):
        """
        训练一个epoch
        
        Args:
            train_loader: 训练数据加载器
            
        Returns:
            float: 平均训练损失
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            # 获取数据
            images = batch['image']
            masks = batch['mask']
            
            # 数据已经是张量，直接使用
            if not isinstance(images, torch.Tensor):
                images, masks = self._pil_to_tensor(images, masks)
            
            # 确保张量维度正确 [B, C, H, W]
            if images.dim() == 4 and images.size(1) != 3:
                images = images.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]
            
            # 确保输入尺寸为512x512
            if images.size(2) != 512 or images.size(3) != 512:
                images = torch.nn.functional.interpolate(images, size=(512, 512), mode='bilinear', align_corners=False)
            
            # 移动到设备
            images = images.to(self.device)
            masks = masks.to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, masks)
            
            # 反向传播
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 100 == 0:
                print(f'Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        return total_loss / max(num_batches, 1)
    
    def validate(self, val_loader):
        """
        验证模型
        
        Args:
            val_loader: 验证数据加载器
            
        Returns:
            float: 平均验证损失
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image']
                masks = batch['mask']
                
                # 数据已经是张量，直接使用
                if not isinstance(images, torch.Tensor):
                    images, masks = self._pil_to_tensor(images, masks)
                
                # 确保张量维度正确 [B, C, H, W]
                if images.dim() == 4 and images.size(1) != 3:
                    images = images.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]
                
                # 确保输入尺寸为512x512
                if images.size(2) != 512 or images.size(3) != 512:
                    images = torch.nn.functional.interpolate(images, size=(512, 512), mode='bilinear', align_corners=False)
                
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / max(num_batches, 1)
    
    def train(self, train_loader, val_loader, num_epochs, save_dir='checkpoints'):
        """
        完整训练过程
        
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            num_epochs: 训练轮数
            save_dir: 模型保存目录
        """
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"开始训练，共{num_epochs}个epoch")
        print(f"设备: {self.device}")
        print(f"模型参数数量: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(num_epochs):
            start_time = time.time()
            
            # 训练
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            
            # 验证
            val_loss = self.validate(val_loader)
            self.val_losses.append(val_loss)
            
            # 更新学习率
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]["lr"]
            self.learning_rates.append(current_lr)
            
            # 记录epoch时间
            epoch_time = time.time() - start_time
            self.epoch_times.append(epoch_time)
            
            # 保存最佳模型
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                torch.save(self.model.state_dict(), 
                          os.path.join(save_dir, 'best_model.pth'))
            
            # 打印进度
            print(f'Epoch {epoch+1}/{num_epochs}:')
            print(f'  训练损失: {train_loss:.4f}')
            print(f'  验证损失: {val_loss:.4f}')
            print(f'  学习率: {current_lr:.6f}')
            print(f'  时间: {epoch_time:.2f}s')
            print('-' * 50)
        
        print(f"训练完成！最佳验证损失: {self.best_val_loss:.4f}")
        
        # 保存实验记录和可视化
        self._save_experiment_results(save_dir)
        self._plot_training_curves(save_dir)
    
    def _save_experiment_results(self, save_dir):
        """保存实验记录到JSON文件"""
        # 更新实验记录
        self.experiment_log.update({
            'end_time': datetime.now().isoformat(),
            'total_epochs': len(self.train_losses),
            'best_val_loss': self.best_val_loss,
            'final_train_loss': self.train_losses[-1] if self.train_losses else None,
            'final_val_loss': self.val_losses[-1] if self.val_losses else None,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'learning_rates': self.learning_rates,
            'epoch_times': self.epoch_times,
            'avg_epoch_time': np.mean(self.epoch_times) if self.epoch_times else 0
        })
        
        # 保存到JSON文件
        log_path = os.path.join(save_dir, 'experiment_log.json')
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(self.experiment_log, f, indent=2, ensure_ascii=False)
        
        print(f"实验记录已保存到: {log_path}")
    
    def _plot_training_curves(self, save_dir):
        """绘制训练曲线"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 损失曲线
        axes[0, 0].plot(self.train_losses, label='训练损失', color='blue')
        axes[0, 0].plot(self.val_losses, label='验证损失', color='red')
        axes[0, 0].set_title('训练和验证损失')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # 学习率曲线
        axes[0, 1].plot(self.learning_rates, color='green')
        axes[0, 1].set_title('学习率变化')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Learning Rate')
        axes[0, 1].grid(True)
        
        # 训练时间曲线
        axes[1, 0].plot(self.epoch_times, color='orange')
        axes[1, 0].set_title('每个Epoch的训练时间')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Time (seconds)')
        axes[1, 0].grid(True)
        
        # 损失对比（对数坐标）
        axes[1, 1].semilogy(self.train_losses, label='训练损失', color='blue')
        axes[1, 1].semilogy(self.val_losses, label='验证损失', color='red')
        axes[1, 1].set_title('损失曲线（对数坐标）')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss (log scale)')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plot_path = os.path.join(save_dir, 'training_curves.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"训练曲线已保存到: {plot_path}")

def main():
    """主函数"""
    print("=== 开始训练 ===")
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建数据加载器
    data_loader = ToothDataLoader(
        data_dir='../data/ToothSegmDataset',
        batch_size=4,  # 小批次，适合测试
        num_workers=2
    )
    data_loader.create_datasets(use_transforms=True)
    data_loader.create_loaders()
    
    train_loader = data_loader.get_train_loader()
    val_loader = data_loader.get_val_loader()
    
    # 创建模型
    model = UNet(num_classes=5).to(device)
    
    # 测试模型输入
    print("测试模型输入...")
    test_input = torch.randn(1, 3, 512, 512).to(device)
    with torch.no_grad():
        test_output = model(test_input)
        print(f"模型测试成功 - 输入: {test_input.shape}, 输出: {test_output.shape}")
    
    # 创建损失函数和优化器
    criterion = CombinedLoss(ce_weight=1.0, dice_weight=1.0)
    optimizer = create_optimizer(model, lr=1e-4)
    scheduler = create_scheduler(optimizer, step_size=10, gamma=0.5)
    
    # 创建训练器
    trainer = Trainer(model, criterion, optimizer, scheduler, device)
    
    # 开始训练
    trainer.train(train_loader, val_loader, num_epochs=5)  # 测试用5个epoch

if __name__ == "__main__":
    main()
