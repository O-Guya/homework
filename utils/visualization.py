"""
可视化工具
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
import cv2
import seaborn as sns
from typing import List, Tuple, Optional
import torch


def show_anns(image, mask, title="Segmentation Result", save_path=None):
    """
    显示分割结果，符合prompt要求
    Args:
        image: 原始RGB图像 (H, W, 3) 或 (3, H, W)
        mask: 分割mask (H, W) 或 (1, H, W)
        title: 图像标题
        save_path: 保存路径
    """
    # 确保图像格式正确
    if len(image.shape) == 3 and image.shape[0] == 3:  # CHW -> HWC
        image = image.transpose(1, 2, 0)
    
    # 确保mask格式正确
    if mask.ndim == 3 and mask.shape[0] == 1:
        mask = mask.squeeze(0)
    
    # 创建图形
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 原始图像
    if image.max() <= 1.0:
        image_display = image
    else:
        image_display = image / 255.0
    
    axes[0].imshow(image_display)
    axes[0].set_title('原始图像')
    axes[0].axis('off')
    
    # 分割mask
    axes[1].imshow(mask, cmap='gray')
    axes[1].set_title('分割Mask')
    axes[1].axis('off')
    
    # 叠加显示
    overlay = image_display.copy()
    mask_binary = (mask > 0.5).astype(np.uint8)
    
    # 创建彩色mask
    colored_mask = np.zeros_like(image_display)
    colored_mask[mask_binary > 0] = [1, 0, 0]  # 红色表示分割区域
    
    # 叠加
    overlay = 0.7 * image_display + 0.3 * colored_mask
    overlay = np.clip(overlay, 0, 1)
    
    axes[2].imshow(overlay)
    axes[2].set_title('叠加显示')
    axes[2].axis('off')
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def save_prediction_visualization(image, true_mask, pred_mask, tooth_id, patient_id, image_id, batch_idx, save_dir):
    """
    保存预测可视化结果
    Args:
        image: 原始图像 (C, H, W)
        true_mask: 真实mask (1, H, W)
        pred_mask: 预测mask (1, H, W)
        tooth_id: 牙齿ID
        patient_id: 患者ID
        image_id: 图像ID
        batch_idx: 批次索引
        save_dir: 保存目录
    """
    # 转换为numpy数组
    if torch.is_tensor(image):
        image = image.cpu().numpy()
    if torch.is_tensor(true_mask):
        true_mask = true_mask.cpu().numpy()
    if torch.is_tensor(pred_mask):
        pred_mask = pred_mask.cpu().numpy()
    
    # 调整图像格式
    if len(image.shape) == 3 and image.shape[0] == 3:  # CHW -> HWC
        image = image.transpose(1, 2, 0)
    
    # 反归一化图像
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = image * std + mean
    image = np.clip(image, 0, 1)
    
    # 调整mask格式
    if true_mask.ndim == 3:
        true_mask = true_mask.squeeze(0)
    if pred_mask.ndim == 3:
        pred_mask = pred_mask.squeeze(0)
    
    # 创建可视化
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 第一行：原始图像、真实mask、预测mask
    axes[0, 0].imshow(image)
    axes[0, 0].set_title(f'原始图像\n牙齿ID: {tooth_id}, 患者: {patient_id}')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(true_mask, cmap='gray')
    axes[0, 1].set_title('真实Mask')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(pred_mask, cmap='gray')
    axes[0, 2].set_title('预测Mask')
    axes[0, 2].axis('off')
    
    # 第二行：叠加显示、差异图、轮廓对比
    # 真实mask叠加
    overlay_true = image.copy()
    true_binary = (true_mask > 0.5).astype(np.uint8)
    overlay_true[true_binary > 0] = [1, 0, 0]  # 红色
    axes[1, 0].imshow(overlay_true)
    axes[1, 0].set_title('真实Mask叠加')
    axes[1, 0].axis('off')
    
    # 预测mask叠加
    overlay_pred = image.copy()
    pred_binary = (pred_mask > 0.5).astype(np.uint8)
    overlay_pred[pred_binary > 0] = [0, 1, 0]  # 绿色
    axes[1, 1].imshow(overlay_pred)
    axes[1, 1].set_title('预测Mask叠加')
    axes[1, 1].axis('off')
    
    # 差异图
    diff = np.abs(true_mask - pred_mask)
    axes[1, 2].imshow(diff, cmap='hot')
    axes[1, 2].set_title('差异图 (越亮差异越大)')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    
    # 保存图像
    save_path = os.path.join(
        save_dir,
        f'tooth_{tooth_id}_patient_{patient_id}_image_{image_id}_batch_{batch_idx}.png'
    )
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_training_curves(train_losses, val_losses, learning_rates, save_path=None):
    """
    绘制训练曲线
    Args:
        train_losses: 训练损失列表
        val_losses: 验证损失列表
        learning_rates: 学习率列表
        save_path: 保存路径
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 损失曲线
    axes[0].plot(train_losses, label='训练损失', color='blue', linewidth=2)
    axes[0].plot(val_losses, label='验证损失', color='red', linewidth=2)
    axes[0].set_title('训练和验证损失', fontsize=14)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 学习率曲线
    axes[1].plot(learning_rates, label='学习率', color='green', linewidth=2)
    axes[1].set_title('学习率调度', fontsize=14)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Learning Rate')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_yscale('log')
    
    # 损失差异
    if len(train_losses) == len(val_losses):
        loss_diff = np.array(val_losses) - np.array(train_losses)
        axes[2].plot(loss_diff, label='验证损失 - 训练损失', color='purple', linewidth=2)
        axes[2].set_title('过拟合指标', fontsize=14)
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Loss Difference')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        axes[2].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_metrics_comparison(metrics_data, save_path=None):
    """
    绘制指标对比图
    Args:
        metrics_data: 指标数据字典
        save_path: 保存路径
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 像素准确率
    if 'pixel_accuracy' in metrics_data:
        axes[0, 0].bar(range(len(metrics_data['pixel_accuracy'])), metrics_data['pixel_accuracy'])
        axes[0, 0].set_title('像素准确率')
        axes[0, 0].set_xlabel('样本')
        axes[0, 0].set_ylabel('Pixel Accuracy')
        axes[0, 0].grid(True, alpha=0.3)
    
    # IoU
    if 'iou' in metrics_data:
        axes[0, 1].bar(range(len(metrics_data['iou'])), metrics_data['iou'])
        axes[0, 1].set_title('IoU')
        axes[0, 1].set_xlabel('样本')
        axes[0, 1].set_ylabel('IoU')
        axes[0, 1].grid(True, alpha=0.3)
    
    # Dice系数
    if 'dice' in metrics_data:
        axes[1, 0].bar(range(len(metrics_data['dice'])), metrics_data['dice'])
        axes[1, 0].set_title('Dice系数')
        axes[1, 0].set_xlabel('样本')
        axes[1, 0].set_ylabel('Dice Coefficient')
        axes[1, 0].grid(True, alpha=0.3)
    
    # 平均指标对比
    if 'mean_metrics' in metrics_data:
        mean_metrics = metrics_data['mean_metrics']
        metric_names = list(mean_metrics.keys())
        metric_values = list(mean_metrics.values())
        
        bars = axes[1, 1].bar(metric_names, metric_values)
        axes[1, 1].set_title('平均指标对比')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 添加数值标签
        for bar, value in zip(bars, metric_values):
            axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def create_confusion_matrix_heatmap(confusion_matrix, class_names=None, save_path=None):
    """
    创建混淆矩阵热力图
    Args:
        confusion_matrix: 混淆矩阵 (numpy array)
        class_names: 类别名称列表
        save_path: 保存路径
    """
    plt.figure(figsize=(8, 6))
    
    if class_names is None:
        class_names = [f'Class {i}' for i in range(confusion_matrix.shape[0])]
    
    sns.heatmap(confusion_matrix, 
                annot=True, 
                fmt='d', 
                cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    
    plt.title('混淆矩阵')
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_tooth_id_performance(tooth_id_results, save_path=None):
    """
    绘制不同牙齿ID的性能对比
    Args:
        tooth_id_results: 牙齿ID结果字典
        save_path: 保存路径
    """
    tooth_ids = list(tooth_id_results.keys())
    metrics = ['pixel_accuracy', 'iou', 'dice']
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for i, metric in enumerate(metrics):
        values = [tooth_id_results[tid][metric] for tid in tooth_ids]
        
        bars = axes[i].bar(range(len(tooth_ids)), values)
        axes[i].set_title(f'{metric.replace("_", " ").title()}')
        axes[i].set_xlabel('牙齿ID')
        axes[i].set_ylabel(metric.replace("_", " ").title())
        axes[i].set_xticks(range(len(tooth_ids)))
        axes[i].set_xticklabels(tooth_ids)
        axes[i].grid(True, alpha=0.3)
        
        # 添加数值标签
        for bar, value in zip(bars, values):
            axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{value:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def visualize_data_samples(dataset, num_samples=8, save_path=None):
    """
    可视化数据样本
    Args:
        dataset: 数据集
        num_samples: 样本数量
        save_path: 保存路径
    """
    fig, axes = plt.subplots(2, num_samples, figsize=(4*num_samples, 8))
    
    for i in range(num_samples):
        sample = dataset[i]
        
        # 原始图像
        image = sample['image']
        if len(image.shape) == 3 and image.shape[0] == 3:  # CHW -> HWC
            image = image.permute(1, 2, 0)
        
        # 反归一化
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = image * std + mean
        image = np.clip(image, 0, 1)
        
        axes[0, i].imshow(image)
        axes[0, i].set_title(f'样本 {i+1}\n牙齿ID: {sample["tooth_id"]}')
        axes[0, i].axis('off')
        
        # 分割mask
        mask = sample['mask']
        if mask.ndim == 3:
            mask = mask.squeeze(0)
        
        axes[1, i].imshow(mask, cmap='gray')
        axes[1, i].set_title('分割Mask')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def create_attention_visualization(model, image, tooth_id, layer_name='film_bottleneck', save_path=None):
    """
    创建注意力可视化（用于调试FiLM层）
    Args:
        model: 模型
        image: 输入图像
        tooth_id: 牙齿ID
        layer_name: 要可视化的层名称
        save_path: 保存路径
    """
    # 注册钩子函数
    activations = {}
    
    def hook_fn(module, input, output):
        activations[layer_name] = output.detach()
    
    # 注册钩子
    hook = None
    for name, module in model.named_modules():
        if name == layer_name:
            hook = module.register_forward_hook(hook_fn)
            break
    
    if hook is None:
        print(f"未找到层: {layer_name}")
        return
    
    # 前向传播
    model.eval()
    with torch.no_grad():
        _ = model(image.unsqueeze(0), torch.tensor([tooth_id]))
    
    # 获取激活
    if layer_name in activations:
        activation = activations[layer_name].squeeze(0)  # 移除batch维度
        
        # 可视化前几个通道
        num_channels = min(8, activation.shape[0])
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()
        
        for i in range(num_channels):
            channel_data = activation[i].cpu().numpy()
            im = axes[i].imshow(channel_data, cmap='viridis')
            axes[i].set_title(f'通道 {i}')
            axes[i].axis('off')
            plt.colorbar(im, ax=axes[i])
        
        # 隐藏多余的子图
        for i in range(num_channels, len(axes)):
            axes[i].axis('off')
        
        plt.suptitle(f'FiLM层激活可视化 - 牙齿ID {tooth_id}')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    # 移除钩子
    hook.remove()


def plot_loss_landscape(model, train_loader, device, save_path=None):
    """
    绘制损失景观（用于调试）
    Args:
        model: 模型
        train_loader: 训练数据加载器
        device: 设备
        save_path: 保存路径
    """
    # 获取一个批次的数据
    batch = next(iter(train_loader))
    images = batch['image'].to(device)
    masks = batch['mask'].to(device)
    tooth_ids = batch['tooth_id'].to(device)
    
    # 计算损失
    model.eval()
    with torch.no_grad():
        outputs = model(images, tooth_ids)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(outputs, masks)
    
    print(f"当前损失: {loss.item():.4f}")
    
    # 这里可以添加更复杂的损失景观分析
    # 例如：参数扰动、学习率扫描等
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


if __name__ == "__main__":
    # 测试可视化函数
    print("可视化工具测试完成")
