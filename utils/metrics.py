"""
评估指标计算
"""
import numpy as np
import torch
import torch.nn.functional as F


def pixel_accuracy(pred, target):
    """
    计算像素准确率
    Args:
        pred: 预测结果 (B, 1, H, W)
        target: 真实标签 (B, 1, H, W)
    Returns:
        accuracy: 像素准确率
    """
    pred = (pred > 0.5).float()
    correct = (pred == target).float()
    accuracy = correct.sum() / correct.numel()
    return accuracy.item()


def mean_pixel_accuracy(pred, target, num_classes=2):
    """
    计算平均像素准确率 (mPA)
    Args:
        pred: 预测结果 (B, 1, H, W)
        target: 真实标签 (B, 1, H, W)
        num_classes: 类别数量
    Returns:
        mpa: 平均像素准确率
    """
    pred = (pred > 0.5).long()
    target = target.long()
    
    # 展平
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    
    # 计算混淆矩阵
    confusion_matrix = torch.zeros(num_classes, num_classes, dtype=torch.long)
    for i in range(num_classes):
        for j in range(num_classes):
            confusion_matrix[i, j] = ((pred_flat == i) & (target_flat == j)).sum()
    
    # 计算每类的准确率
    class_acc = confusion_matrix.diag() / confusion_matrix.sum(dim=1).float()
    class_acc = class_acc[~torch.isnan(class_acc)]  # 移除NaN值
    
    mpa = class_acc.mean().item()
    return mpa


def intersection_over_union(pred, target, smooth=1e-6):
    """
    计算IoU
    Args:
        pred: 预测结果 (B, 1, H, W)
        target: 真实标签 (B, 1, H, W)
        smooth: 平滑因子
    Returns:
        iou: IoU值
    """
    pred = (pred > 0.5).float()
    target = target.float()
    
    # 计算交集和并集
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    
    iou = (intersection + smooth) / (union + smooth)
    return iou.item()


def mean_intersection_over_union(pred, target, num_classes=2, smooth=1e-6):
    """
    计算平均IoU (mIoU)
    Args:
        pred: 预测结果 (B, 1, H, W)
        target: 真实标签 (B, 1, H, W)
        num_classes: 类别数量
        smooth: 平滑因子
    Returns:
        miou: 平均IoU
    """
    pred = (pred > 0.5).long()
    target = target.long()
    
    # 展平
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    
    # 计算混淆矩阵
    confusion_matrix = torch.zeros(num_classes, num_classes, dtype=torch.long)
    for i in range(num_classes):
        for j in range(num_classes):
            confusion_matrix[i, j] = ((pred_flat == i) & (target_flat == j)).sum()
    
    # 计算每类的IoU
    intersection = confusion_matrix.diag()
    union = confusion_matrix.sum(dim=1) + confusion_matrix.sum(dim=0) - intersection
    
    iou = (intersection.float() + smooth) / (union.float() + smooth)
    iou = iou[~torch.isnan(iou)]  # 移除NaN值
    
    miou = iou.mean().item()
    return miou


def dice_coefficient(pred, target, smooth=1e-6):
    """
    计算Dice系数
    Args:
        pred: 预测结果 (B, 1, H, W)
        target: 真实标签 (B, 1, H, W)
        smooth: 平滑因子
    Returns:
        dice: Dice系数
    """
    pred = (pred > 0.5).float()
    target = target.float()
    
    intersection = (pred * target).sum()
    dice = (2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    return dice.item()


def calculate_metrics(pred, target):
    """
    计算所有评估指标
    Args:
        pred: 预测结果 (B, 1, H, W)
        target: 真实标签 (B, 1, H, W)
    Returns:
        metrics: 包含所有指标的字典
    """
    metrics = {}
    
    # 像素准确率
    metrics['pixel_accuracy'] = pixel_accuracy(pred, target)
    
    # 平均像素准确率
    metrics['mean_pixel_accuracy'] = mean_pixel_accuracy(pred, target)
    
    # IoU
    metrics['iou'] = intersection_over_union(pred, target)
    
    # 平均IoU
    metrics['mean_iou'] = mean_intersection_over_union(pred, target)
    
    # Dice系数
    metrics['dice'] = dice_coefficient(pred, target)
    
    return metrics


class MetricsTracker:
    """
    指标跟踪器
    """
    def __init__(self):
        self.reset()
    
    def reset(self):
        """重置所有指标"""
        self.pixel_accuracy = []
        self.mean_pixel_accuracy = []
        self.iou = []
        self.mean_iou = []
        self.dice = []
    
    def update(self, pred, target):
        """更新指标"""
        metrics = calculate_metrics(pred, target)
        self.pixel_accuracy.append(metrics['pixel_accuracy'])
        self.mean_pixel_accuracy.append(metrics['mean_pixel_accuracy'])
        self.iou.append(metrics['iou'])
        self.mean_iou.append(metrics['mean_iou'])
        self.dice.append(metrics['dice'])
    
    def get_average_metrics(self):
        """获取平均指标"""
        return {
            'pixel_accuracy': np.mean(self.pixel_accuracy),
            'mean_pixel_accuracy': np.mean(self.mean_pixel_accuracy),
            'iou': np.mean(self.iou),
            'mean_iou': np.mean(self.mean_iou),
            'dice': np.mean(self.dice)
        }
    
    def get_std_metrics(self):
        """获取指标标准差"""
        return {
            'pixel_accuracy': np.std(self.pixel_accuracy),
            'mean_pixel_accuracy': np.std(self.mean_pixel_accuracy),
            'iou': np.std(self.iou),
            'mean_iou': np.std(self.mean_iou),
            'dice': np.std(self.dice)
        }
