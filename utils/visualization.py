"""
可视化工具
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from typing import Optional, List, Tuple
import cv2


def denormalize_image(image: torch.Tensor, mean: List[float] = [0.485, 0.456, 0.406], 
                     std: List[float] = [0.229, 0.224, 0.225]) -> np.ndarray:
    """
    反归一化图像
    
    Args:
        image: 归一化的图像张量 (C, H, W)
        mean: 归一化均值
        std: 归一化标准差
    
    Returns:
        反归一化后的图像数组 (H, W, C)
    """
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
    
    # 转换为 (H, W, C)
    if image.shape[0] == 3:
        image = np.transpose(image, (1, 2, 0))
    
    # 反归一化
    image = image * np.array(std) + np.array(mean)
    image = np.clip(image, 0, 1)
    
    return image


def save_predictions(images: torch.Tensor, masks: torch.Tensor, predictions: torch.Tensor, 
                    save_path: str, num_samples: int = 4):
    """
    保存预测结果可视化
    
    Args:
        images: 输入图像 (B, C, H, W)
        masks: 真实掩码 (B, H, W)
        predictions: 预测掩码 (B, H, W)
        save_path: 保存路径
        num_samples: 显示的样本数量
    """
    # 转换为numpy
    if isinstance(images, torch.Tensor):
        images = images.cpu().numpy()
    if isinstance(masks, torch.Tensor):
        masks = masks.cpu().numpy()
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    
    # 限制样本数量
    num_samples = min(num_samples, len(images))
    
    # 创建子图
    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_samples):
        # 原始图像
        img = denormalize_image(images[i])
        axes[i, 0].imshow(img)
        axes[i, 0].set_title('Original Image')
        axes[i, 0].axis('off')
        
        # 真实掩码
        mask = masks[i]
        axes[i, 1].imshow(mask, cmap='gray')
        axes[i, 1].set_title('Ground Truth')
        axes[i, 1].axis('off')
        
        # 预测掩码
        pred = predictions[i]
        axes[i, 2].imshow(pred, cmap='gray')
        axes[i, 2].set_title('Prediction')
        axes[i, 2].axis('off')
        
        # 叠加显示
        overlay = img.copy()
        # 将预测结果叠加到原图上（红色表示预测区域）
        overlay[pred == 1] = [1.0, 0.0, 0.0]  # 红色
        axes[i, 3].imshow(overlay)
        axes[i, 3].set_title('Overlay')
        axes[i, 3].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_training_curves(metrics: dict, save_path: str):
    """
    绘制训练曲线
    
    Args:
        metrics: 包含训练指标的字典
        save_path: 保存路径
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # 损失曲线
    axes[0, 0].plot(metrics['train_loss'], label='Train Loss')
    axes[0, 0].plot(metrics['val_loss'], label='Val Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Dice分数曲线
    axes[0, 1].plot(metrics['val_dice'], label='Val Dice', color='green')
    axes[0, 1].set_title('Validation Dice Score')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Dice Score')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # IoU分数曲线
    axes[1, 0].plot(metrics['val_iou'], label='Val IoU', color='orange')
    axes[1, 0].set_title('Validation IoU Score')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('IoU Score')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # 学习率曲线（如果有的话）
    if 'learning_rate' in metrics:
        axes[1, 1].plot(metrics['learning_rate'], label='Learning Rate', color='red')
        axes[1, 1].set_title('Learning Rate Schedule')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
    else:
        axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_confusion_matrix_plot(predictions: np.ndarray, targets: np.ndarray, 
                                save_path: str, class_names: List[str] = ['Background', 'Tooth']):
    """
    创建混淆矩阵图
    
    Args:
        predictions: 预测标签
        targets: 真实标签
        save_path: 保存路径
        class_names: 类别名称
    """
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    
    # 计算混淆矩阵
    cm = confusion_matrix(targets.flatten(), predictions.flatten())
    
    # 绘制混淆矩阵
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def visualize_data_distribution(dataset, save_path: str, num_samples: int = 16):
    """
    可视化数据分布
    
    Args:
        dataset: 数据集对象
        save_path: 保存路径
        num_samples: 显示的样本数量
    """
    num_samples = min(num_samples, len(dataset))
    
    # 创建子图
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    axes = axes.flatten()
    
    for i in range(num_samples):
        sample = dataset[i]
        image = denormalize_image(sample['image'])
        
        axes[i].imshow(image)
        axes[i].set_title(f"Tooth Type: {sample['tooth_type']}")
        axes[i].axis('off')
    
    # 隐藏多余的子图
    for i in range(num_samples, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_attention_visualization(model, image: torch.Tensor, layer_name: str, 
                                 save_path: str):
    """
    创建注意力可视化
    
    Args:
        model: 模型
        image: 输入图像
        layer_name: 要可视化的层名称
        save_path: 保存路径
    """
    # 注册钩子函数
    activations = {}
    
    def hook_fn(module, input, output):
        activations[layer_name] = output.detach()
    
    # 注册钩子
    for name, module in model.named_modules():
        if name == layer_name:
            hook = module.register_forward_hook(hook_fn)
            break
    
    # 前向传播
    model.eval()
    with torch.no_grad():
        _ = model(image.unsqueeze(0))
    
    # 移除钩子
    hook.remove()
    
    # 获取激活
    if layer_name in activations:
        activation = activations[layer_name].squeeze(0)  # 移除batch维度
        
        # 计算平均激活
        avg_activation = torch.mean(activation, dim=0)
        
        # 可视化
        plt.figure(figsize=(10, 8))
        plt.imshow(avg_activation.cpu().numpy(), cmap='hot')
        plt.colorbar()
        plt.title(f'Attention Map - {layer_name}')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()


def save_model_architecture(model, save_path: str, input_size: Tuple[int, int, int] = (3, 512, 512)):
    """
    保存模型架构图
    
    Args:
        model: 模型
        save_path: 保存路径
        input_size: 输入尺寸 (C, H, W)
    """
    try:
        from torchsummary import summary
        from io import StringIO
        import sys
        
        # 重定向输出
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        
        # 生成模型摘要
        summary(model, input_size)
        
        # 获取输出
        model_summary = sys.stdout.getvalue()
        sys.stdout = old_stdout
        
        # 保存到文件
        with open(save_path, 'w') as f:
            f.write(model_summary)
            
    except ImportError:
        print("torchsummary not available, skipping model architecture visualization")


if __name__ == "__main__":
    # 测试可视化函数
    print("=== 测试可视化函数 ===")
    
    # 创建测试数据
    batch_size = 4
    height, width = 256, 256
    
    # 创建测试图像和掩码
    images = torch.randn(batch_size, 3, height, width)
    masks = torch.randint(0, 2, (batch_size, height, width))
    predictions = torch.randint(0, 2, (batch_size, height, width))
    
    # 测试保存预测结果
    save_predictions(images, masks, predictions, 'test_predictions.png')
    print("✅ 预测结果可视化测试完成")
    
    # 测试训练曲线
    metrics = {
        'train_loss': [0.8, 0.6, 0.4, 0.3, 0.2],
        'val_loss': [0.9, 0.7, 0.5, 0.4, 0.3],
        'val_dice': [0.1, 0.3, 0.5, 0.7, 0.8],
        'val_iou': [0.1, 0.2, 0.4, 0.6, 0.7]
    }
    
    plot_training_curves(metrics, 'test_training_curves.png')
    print("✅ 训练曲线可视化测试完成")
    
    print("✅ 所有可视化测试完成！")
