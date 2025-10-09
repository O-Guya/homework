"""
训练脚本 - 牙齿分割模型
"""
import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from config import *
from models.unet import create_model
from models.loss import create_loss_function
from utils.dataset import create_data_loaders
from utils.metrics import MetricsTracker


class EarlyStopping:
    """早停机制"""
    def __init__(self, patience=7, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1

        if self.counter >= self.patience:
            if self.restore_best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False

    def save_checkpoint(self, model):
        self.best_weights = model.state_dict().copy()


class Trainer:
    """训练器类"""
    def __init__(self, model, train_loader, val_loader, device, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config
        
        # 损失函数
        self.criterion = create_loss_function(
            loss_type='combined',
            bce_weight=1.0,
            dice_weight=1.0,
            focal_weight=0.5
        )
        
        # 优化器
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        # 早停
        self.early_stopping = EarlyStopping(
            patience=config['patience'],
            min_delta=1e-4
        )
        
        # 指标跟踪器
        self.train_metrics = MetricsTracker()
        self.val_metrics = MetricsTracker()
        
        # TensorBoard
        self.writer = SummaryWriter(LOG_SAVE_PATH)
        
        # 训练历史
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []

    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        self.train_metrics.reset()
        
        total_loss = 0.0
        progress_bar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.config["num_epochs"]}')
        
        for batch_idx, batch in enumerate(progress_bar):
            # 获取数据
            images = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device)
            tooth_ids = batch['tooth_id'].to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            outputs = self.model(images, tooth_ids)
            
            # 计算损失
            loss = self.criterion(outputs, masks)
            
            # 反向传播
            loss.backward()
            self.optimizer.step()
            
            # 更新指标
            with torch.no_grad():
                pred_sigmoid = torch.sigmoid(outputs)
                self.train_metrics.update(pred_sigmoid, masks)
            
            total_loss += loss.item()
            
            # 更新进度条
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Avg Loss': f'{total_loss/(batch_idx+1):.4f}'
            })
            
            # 记录到TensorBoard
            global_step = epoch * len(self.train_loader) + batch_idx
            self.writer.add_scalar('Train/Loss', loss.item(), global_step)
        
        avg_loss = total_loss / len(self.train_loader)
        avg_metrics = self.train_metrics.get_average_metrics()
        
        return avg_loss, avg_metrics

    def validate_epoch(self, epoch):
        """验证一个epoch"""
        self.model.eval()
        self.val_metrics.reset()
        
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                # 获取数据
                images = batch['image'].to(self.device)
                masks = batch['mask'].to(self.device)
                tooth_ids = batch['tooth_id'].to(self.device)
                
                # 前向传播
                outputs = self.model(images, tooth_ids)
                
                # 计算损失
                loss = self.criterion(outputs, masks)
                total_loss += loss.item()
                
                # 更新指标
                pred_sigmoid = torch.sigmoid(outputs)
                self.val_metrics.update(pred_sigmoid, masks)
        
        avg_loss = total_loss / len(self.val_loader)
        avg_metrics = self.val_metrics.get_average_metrics()
        
        return avg_loss, avg_metrics

    def train(self):
        """完整训练过程"""
        print("开始训练...")
        print(f"设备: {self.device}")
        print(f"模型参数数量: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"训练样本数: {len(self.train_loader.dataset)}")
        print(f"验证样本数: {len(self.val_loader.dataset)}")
        
        best_val_loss = float('inf')
        start_time = time.time()
        
        for epoch in range(self.config['num_epochs']):
            epoch_start_time = time.time()
            
            # 训练
            train_loss, train_metrics = self.train_epoch(epoch)
            
            # 验证
            val_loss, val_metrics = self.validate_epoch(epoch)
            
            # 学习率调度
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # 记录历史
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.learning_rates.append(current_lr)
            
            # 记录到TensorBoard
            self.writer.add_scalar('Epoch/Train_Loss', train_loss, epoch)
            self.writer.add_scalar('Epoch/Val_Loss', val_loss, epoch)
            self.writer.add_scalar('Epoch/Learning_Rate', current_lr, epoch)
            self.writer.add_scalar('Epoch/Train_Pixel_Accuracy', train_metrics['pixel_accuracy'], epoch)
            self.writer.add_scalar('Epoch/Val_Pixel_Accuracy', val_metrics['pixel_accuracy'], epoch)
            self.writer.add_scalar('Epoch/Train_IoU', train_metrics['iou'], epoch)
            self.writer.add_scalar('Epoch/Val_IoU', val_metrics['iou'], epoch)
            self.writer.add_scalar('Epoch/Train_Dice', train_metrics['dice'], epoch)
            self.writer.add_scalar('Epoch/Val_Dice', val_metrics['dice'], epoch)
            
            # 打印epoch结果
            epoch_time = time.time() - epoch_start_time
            print(f"\nEpoch {epoch+1}/{self.config['num_epochs']} - {epoch_time:.2f}s")
            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            print(f"Train PA: {train_metrics['pixel_accuracy']:.4f}, Val PA: {val_metrics['pixel_accuracy']:.4f}")
            print(f"Train IoU: {train_metrics['iou']:.4f}, Val IoU: {val_metrics['iou']:.4f}")
            print(f"Train Dice: {train_metrics['dice']:.4f}, Val Dice: {val_metrics['dice']:.4f}")
            print(f"Learning Rate: {current_lr:.6f}")
            
            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_metrics': val_metrics,
                    'config': self.config
                }, os.path.join(MODEL_SAVE_PATH, 'best_model.pth'))
                print(f"保存最佳模型 (Val Loss: {val_loss:.4f})")
            
            # 早停检查
            if self.early_stopping(val_loss, self.model):
                print(f"早停触发，在第 {epoch+1} 轮停止训练")
                break
        
        # 训练完成
        total_time = time.time() - start_time
        print(f"\n训练完成！总用时: {total_time/3600:.2f} 小时")
        
        # 保存最终模型
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'val_metrics': val_metrics,
            'config': self.config,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'learning_rates': self.learning_rates
        }, os.path.join(MODEL_SAVE_PATH, 'final_model.pth'))
        
        # 绘制训练曲线
        self.plot_training_curves()
        
        # 关闭TensorBoard
        self.writer.close()

    def plot_training_curves(self):
        """绘制训练曲线"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 损失曲线
        axes[0, 0].plot(self.train_losses, label='Train Loss', color='blue')
        axes[0, 0].plot(self.val_losses, label='Val Loss', color='red')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # 学习率曲线
        axes[0, 1].plot(self.learning_rates, label='Learning Rate', color='green')
        axes[0, 1].set_title('Learning Rate Schedule')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Learning Rate')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        axes[0, 1].set_yscale('log')
        
        # 像素准确率曲线
        train_pa = [m['pixel_accuracy'] for m in self.train_metrics.get_average_metrics()]
        val_pa = [m['pixel_accuracy'] for m in self.val_metrics.get_average_metrics()]
        axes[1, 0].plot(train_pa, label='Train PA', color='blue')
        axes[1, 0].plot(val_pa, label='Val PA', color='red')
        axes[1, 0].set_title('Pixel Accuracy')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Pixel Accuracy')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # IoU曲线
        train_iou = [m['iou'] for m in self.train_metrics.get_average_metrics()]
        val_iou = [m['iou'] for m in self.val_metrics.get_average_metrics()]
        axes[1, 1].plot(train_iou, label='Train IoU', color='blue')
        axes[1, 1].plot(val_iou, label='Val IoU', color='red')
        axes[1, 1].set_title('Intersection over Union')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('IoU')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(RESULT_SAVE_PATH, 'training_curves.png'), dpi=300, bbox_inches='tight')
        plt.show()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='牙齿分割模型训练')
    parser.add_argument('--batch_size', type=int, default=TRAIN_CONFIG['batch_size'], help='批次大小')
    parser.add_argument('--epochs', type=int, default=TRAIN_CONFIG['num_epochs'], help='训练轮数')
    parser.add_argument('--lr', type=float, default=TRAIN_CONFIG['learning_rate'], help='学习率')
    parser.add_argument('--tooth_ids', nargs='+', type=int, default=TRAIN_CONFIG['selected_tooth_ids'], 
                       help='选择的牙齿ID列表')
    parser.add_argument('--device', type=str, default='auto', help='设备 (cuda/cpu/auto)')
    parser.add_argument('--resume', type=str, default=None, help='恢复训练的模型路径')
    
    args = parser.parse_args()
    
    # 设备选择
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"使用设备: {device}")
    
    # 更新配置
    config = TRAIN_CONFIG.copy()
    config.update({
        'batch_size': args.batch_size,
        'num_epochs': args.epochs,
        'learning_rate': args.lr,
        'selected_tooth_ids': args.tooth_ids
    })
    
    print(f"训练配置: {config}")
    print(f"选择的牙齿ID: {args.tooth_ids}")
    
    # 创建数据加载器
    print("创建数据加载器...")
    train_loader, val_loader = create_data_loaders(
        selected_tooth_ids=args.tooth_ids,
        batch_size=config['batch_size'],
        val_split=config['val_split']
    )
    
    # 创建模型
    print("创建模型...")
    model = create_model(MODEL_CONFIG)
    model = model.to(device)
    
    # 创建训练器
    trainer = Trainer(model, train_loader, val_loader, device, config)
    
    # 恢复训练（如果指定）
    if args.resume:
        print(f"恢复训练: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"从第 {checkpoint['epoch']+1} 轮恢复训练")
    
    # 开始训练
    trainer.train()


if __name__ == "__main__":
    main()
