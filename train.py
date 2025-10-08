"""
牙齿分割训练脚本
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import os
import time
import json
from pathlib import Path
import logging
from tqdm import tqdm
import numpy as np

from models.unet import UNet
from data_loader import create_dataloaders
from utils.metrics import DiceLoss, IoUScore, DiceScore
from utils.visualization import save_predictions

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ToothSegmentationTrainer:
    """牙齿分割训练器"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"使用设备: {self.device}")
        
        # 创建输出目录
        self.output_dir = Path(config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存配置
        with open(self.output_dir / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)
        
        # 初始化模型
        self.model = UNet(num_classes=config['num_classes']).to(self.device)
        logger.info(f"模型参数数量: {sum(p.numel() for p in self.model.parameters()):,}")
        
        # 初始化损失函数
        self.criterion = DiceLoss()
        self.aux_criterion = nn.CrossEntropyLoss()
        
        # 初始化优化器
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=config['patience']
        )
        
        # 初始化数据加载器
        self.train_loader, self.val_loader, self.test_loader = create_dataloaders(
            data_dir=config['data_dir'],
            batch_size=config['batch_size'],
            num_workers=config['num_workers'],
            image_size=config['image_size'],
            tooth_types=config.get('tooth_types', None),
            train_augment=True
        )
        
        # 初始化指标
        self.metrics = {
            'train_loss': [],
            'val_loss': [],
            'val_dice': [],
            'val_iou': []
        }
        
        # 初始化TensorBoard
        self.writer = SummaryWriter(self.output_dir / 'logs')
        
        # 最佳模型跟踪
        self.best_dice = 0.0
        self.epochs_without_improvement = 0
        
    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.config["epochs"]}')
        
        for batch_idx, batch in enumerate(pbar):
            images = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            outputs = self.model(images)
            
            # 计算损失
            dice_loss = self.criterion(outputs, masks)
            ce_loss = self.aux_criterion(outputs, masks)
            total_loss_batch = dice_loss + 0.5 * ce_loss
            
            # 反向传播
            total_loss_batch.backward()
            self.optimizer.step()
            
            total_loss += total_loss_batch.item()
            
            # 更新进度条
            pbar.set_postfix({
                'Loss': f'{total_loss_batch.item():.4f}',
                'Avg Loss': f'{total_loss / (batch_idx + 1):.4f}'
            })
        
        avg_loss = total_loss / num_batches
        self.metrics['train_loss'].append(avg_loss)
        
        return avg_loss
    
    def validate_epoch(self, epoch):
        """验证一个epoch"""
        self.model.eval()
        total_loss = 0.0
        total_dice = 0.0
        total_iou = 0.0
        num_batches = len(self.val_loader)
        
        dice_metric = DiceScore()
        iou_metric = IoUScore()
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                images = batch['image'].to(self.device)
                masks = batch['mask'].to(self.device)
                
                # 前向传播
                outputs = self.model(images)
                
                # 计算损失
                dice_loss = self.criterion(outputs, masks)
                ce_loss = self.aux_criterion(outputs, masks)
                total_loss_batch = dice_loss + 0.5 * ce_loss
                
                total_loss += total_loss_batch.item()
                
                # 计算指标
                predictions = torch.argmax(outputs, dim=1)
                dice_score = dice_metric(predictions, masks)
                iou_score = iou_metric(predictions, masks)
                
                total_dice += dice_score
                total_iou += iou_score
        
        avg_loss = total_loss / num_batches
        avg_dice = total_dice / num_batches
        avg_iou = total_iou / num_batches
        
        self.metrics['val_loss'].append(avg_loss)
        self.metrics['val_dice'].append(avg_dice)
        self.metrics['val_iou'].append(avg_iou)
        
        return avg_loss, avg_dice, avg_iou
    
    def save_checkpoint(self, epoch, is_best=False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_dice': self.best_dice,
            'metrics': self.metrics
        }
        
        # 保存最新检查点
        torch.save(checkpoint, self.output_dir / 'checkpoint_latest.pth')
        
        # 保存最佳模型
        if is_best:
            torch.save(checkpoint, self.output_dir / 'checkpoint_best.pth')
            logger.info(f"保存最佳模型 (Dice: {self.best_dice:.4f})")
    
    def train(self):
        """主训练循环"""
        logger.info("开始训练...")
        start_time = time.time()
        
        for epoch in range(self.config['epochs']):
            # 训练
            train_loss = self.train_epoch(epoch)
            
            # 验证
            val_loss, val_dice, val_iou = self.validate_epoch(epoch)
            
            # 更新学习率
            self.scheduler.step(val_dice)
            
            # 记录到TensorBoard
            self.writer.add_scalar('Loss/Train', train_loss, epoch)
            self.writer.add_scalar('Loss/Val', val_loss, epoch)
            self.writer.add_scalar('Metrics/Dice', val_dice, epoch)
            self.writer.add_scalar('Metrics/IoU', val_iou, epoch)
            self.writer.add_scalar('Learning_Rate', self.optimizer.param_groups[0]['lr'], epoch)
            
            # 打印结果
            logger.info(
                f"Epoch {epoch+1}/{self.config['epochs']}: "
                f"Train Loss: {train_loss:.4f}, "
                f"Val Loss: {val_loss:.4f}, "
                f"Val Dice: {val_dice:.4f}, "
                f"Val IoU: {val_iou:.4f}"
            )
            
            # 检查是否是最佳模型
            is_best = val_dice > self.best_dice
            if is_best:
                self.best_dice = val_dice
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1
            
            # 保存检查点
            self.save_checkpoint(epoch, is_best)
            
            # 早停
            if self.epochs_without_improvement >= self.config['early_stopping_patience']:
                logger.info(f"早停触发，{self.config['early_stopping_patience']} 个epoch无改善")
                break
        
        # 训练完成
        total_time = time.time() - start_time
        logger.info(f"训练完成！总时间: {total_time/3600:.2f} 小时")
        logger.info(f"最佳Dice分数: {self.best_dice:.4f}")
        
        # 保存最终模型
        torch.save(self.model.state_dict(), self.output_dir / 'model_final.pth')
        
        # 关闭TensorBoard
        self.writer.close()
    
    def test(self):
        """测试模型"""
        logger.info("开始测试...")
        
        # 加载最佳模型
        checkpoint = torch.load(self.output_dir / 'checkpoint_best.pth')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        self.model.eval()
        test_dice = 0.0
        test_iou = 0.0
        num_batches = len(self.test_loader)
        
        dice_metric = DiceScore()
        iou_metric = IoUScore()
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.test_loader, desc='Testing')):
                images = batch['image'].to(self.device)
                masks = batch['mask'].to(self.device)
                
                # 前向传播
                outputs = self.model(images)
                predictions = torch.argmax(outputs, dim=1)
                
                # 计算指标
                dice_score = dice_metric(predictions, masks)
                iou_score = iou_metric(predictions, masks)
                
                test_dice += dice_score
                test_iou += iou_score
                
                # 保存一些预测结果用于可视化
                if batch_idx < 5:  # 只保存前5个批次
                    save_predictions(
                        images, masks, predictions,
                        self.output_dir / f'test_predictions_batch_{batch_idx}.png'
                    )
        
        avg_dice = test_dice / num_batches
        avg_iou = test_iou / num_batches
        
        logger.info(f"测试结果 - Dice: {avg_dice:.4f}, IoU: {avg_iou:.4f}")
        
        # 保存测试结果
        test_results = {
            'dice': avg_dice,
            'iou': avg_iou,
            'num_batches': num_batches
        }
        
        with open(self.output_dir / 'test_results.json', 'w') as f:
            json.dump(test_results, f, indent=2)


def main():
    """主函数"""
    # 配置
    config = {
        'data_dir': '/mnt/external_4tb/hjy/assignment1/data/ToothSegmDataset',
        'output_dir': '/mnt/external_4tb/hjy/assignment1/results',
        'num_classes': 2,  # 背景 + 牙齿
        'image_size': 512,
        'batch_size': 8,
        'num_workers': 0,  # 暂时使用单进程避免多进程问题
        'epochs': 100,
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'patience': 10,
        'early_stopping_patience': 20,
        'tooth_types': None  # 使用所有牙齿类型
    }
    
    # 创建训练器
    trainer = ToothSegmentationTrainer(config)
    
    # 训练
    trainer.train()
    
    # 测试
    trainer.test()


if __name__ == "__main__":
    main()
