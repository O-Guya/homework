"""
分割任务评估指标
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional


class DiceLoss(nn.Module):
    """Dice损失函数"""
    
    def __init__(self, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, predictions, targets):
        """
        计算Dice损失
        
        Args:
            predictions: 预测logits (B, C, H, W)
            targets: 真实标签 (B, H, W)
        
        Returns:
            Dice损失
        """
        # 将预测转换为概率
        predictions = F.softmax(predictions, dim=1)
        
        # 获取前景类别（类别1）
        pred_foreground = predictions[:, 1, :, :]  # (B, H, W)
        target_foreground = (targets == 1).float()  # (B, H, W)
        
        # 计算Dice系数
        intersection = (pred_foreground * target_foreground).sum(dim=(1, 2))
        union = pred_foreground.sum(dim=(1, 2)) + target_foreground.sum(dim=(1, 2))
        
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        
        # 返回1 - Dice作为损失
        return 1 - dice.mean()


class DiceScore:
    """Dice分数计算器"""
    
    def __init__(self, smooth=1e-5):
        self.smooth = smooth
    
    def __call__(self, predictions, targets):
        """
        计算Dice分数
        
        Args:
            predictions: 预测标签 (B, H, W)
            targets: 真实标签 (B, H, W)
        
        Returns:
            Dice分数
        """
        # 转换为numpy
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.cpu().numpy()
        
        # 计算每个样本的Dice分数
        dice_scores = []
        for pred, target in zip(predictions, targets):
            # 二值化
            pred_binary = (pred == 1).astype(np.float32)
            target_binary = (target == 1).astype(np.float32)
            
            # 计算Dice
            intersection = np.sum(pred_binary * target_binary)
            union = np.sum(pred_binary) + np.sum(target_binary)
            
            if union == 0:
                dice = 1.0 if intersection == 0 else 0.0
            else:
                dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
            
            dice_scores.append(dice)
        
        return np.mean(dice_scores)


class IoUScore:
    """IoU分数计算器"""
    
    def __init__(self, smooth=1e-5):
        self.smooth = smooth
    
    def __call__(self, predictions, targets):
        """
        计算IoU分数
        
        Args:
            predictions: 预测标签 (B, H, W)
            targets: 真实标签 (B, H, W)
        
        Returns:
            IoU分数
        """
        # 转换为numpy
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.cpu().numpy()
        
        # 计算每个样本的IoU分数
        iou_scores = []
        for pred, target in zip(predictions, targets):
            # 二值化
            pred_binary = (pred == 1).astype(np.float32)
            target_binary = (target == 1).astype(np.float32)
            
            # 计算IoU
            intersection = np.sum(pred_binary * target_binary)
            union = np.sum(pred_binary) + np.sum(target_binary) - intersection
            
            if union == 0:
                iou = 1.0 if intersection == 0 else 0.0
            else:
                iou = (intersection + self.smooth) / (union + self.smooth)
            
            iou_scores.append(iou)
        
        return np.mean(iou_scores)


class PixelAccuracy:
    """像素准确率计算器"""
    
    def __call__(self, predictions, targets):
        """
        计算像素准确率
        
        Args:
            predictions: 预测标签 (B, H, W)
            targets: 真实标签 (B, H, W)
        
        Returns:
            像素准确率
        """
        # 转换为numpy
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.cpu().numpy()
        
        # 计算准确率
        correct = (predictions == targets).sum()
        total = predictions.size
        
        return correct / total


class PrecisionScore:
    """精确率计算器"""
    
    def __call__(self, predictions, targets):
        """
        计算精确率
        
        Args:
            predictions: 预测标签 (B, H, W)
            targets: 真实标签 (B, H, W)
        
        Returns:
            精确率
        """
        # 转换为numpy
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.cpu().numpy()
        
        # 二值化
        pred_binary = (predictions == 1).astype(np.float32)
        target_binary = (targets == 1).astype(np.float32)
        
        # 计算精确率
        true_positive = np.sum(pred_binary * target_binary)
        false_positive = np.sum(pred_binary * (1 - target_binary))
        
        if true_positive + false_positive == 0:
            return 0.0
        
        return true_positive / (true_positive + false_positive)


class RecallScore:
    """召回率计算器"""
    
    def __call__(self, predictions, targets):
        """
        计算召回率
        
        Args:
            predictions: 预测标签 (B, H, W)
            targets: 真实标签 (B, H, W)
        
        Returns:
            召回率
        """
        # 转换为numpy
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.cpu().numpy()
        
        # 二值化
        pred_binary = (predictions == 1).astype(np.float32)
        target_binary = (targets == 1).astype(np.float32)
        
        # 计算召回率
        true_positive = np.sum(pred_binary * target_binary)
        false_negative = np.sum((1 - pred_binary) * target_binary)
        
        if true_positive + false_negative == 0:
            return 0.0
        
        return true_positive / (true_positive + false_negative)


class F1Score:
    """F1分数计算器"""
    
    def __call__(self, predictions, targets):
        """
        计算F1分数
        
        Args:
            predictions: 预测标签 (B, H, W)
            targets: 真实标签 (B, H, W)
        
        Returns:
            F1分数
        """
        precision = PrecisionScore()(predictions, targets)
        recall = RecallScore()(predictions, targets)
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * (precision * recall) / (precision + recall)


def compute_all_metrics(predictions, targets):
    """
    计算所有指标
    
    Args:
        predictions: 预测标签 (B, H, W)
        targets: 真实标签 (B, H, W)
    
    Returns:
        包含所有指标的字典
    """
    metrics = {}
    
    # 计算各种指标
    metrics['dice'] = DiceScore()(predictions, targets)
    metrics['iou'] = IoUScore()(predictions, targets)
    metrics['pixel_accuracy'] = PixelAccuracy()(predictions, targets)
    metrics['precision'] = PrecisionScore()(predictions, targets)
    metrics['recall'] = RecallScore()(predictions, targets)
    metrics['f1'] = F1Score()(predictions, targets)
    
    return metrics


if __name__ == "__main__":
    # 测试指标
    print("=== 测试分割指标 ===")
    
    # 创建测试数据
    batch_size = 4
    height, width = 256, 256
    
    # 创建预测和真实标签
    predictions = torch.randint(0, 2, (batch_size, height, width))
    targets = torch.randint(0, 2, (batch_size, height, width))
    
    print(f"预测形状: {predictions.shape}")
    print(f"真实标签形状: {targets.shape}")
    
    # 测试各种指标
    dice_score = DiceScore()(predictions, targets)
    iou_score = IoUScore()(predictions, targets)
    pixel_acc = PixelAccuracy()(predictions, targets)
    precision = PrecisionScore()(predictions, targets)
    recall = RecallScore()(predictions, targets)
    f1 = F1Score()(predictions, targets)
    
    print(f"Dice分数: {dice_score:.4f}")
    print(f"IoU分数: {iou_score:.4f}")
    print(f"像素准确率: {pixel_acc:.4f}")
    print(f"精确率: {precision:.4f}")
    print(f"召回率: {recall:.4f}")
    print(f"F1分数: {f1:.4f}")
    
    # 测试所有指标
    all_metrics = compute_all_metrics(predictions, targets)
    print(f"\n所有指标: {all_metrics}")
    
    # 测试Dice损失
    predictions_logits = torch.randn(batch_size, 2, height, width)
    dice_loss = DiceLoss()(predictions_logits, targets)
    print(f"Dice损失: {dice_loss:.4f}")
    
    print("✅ 指标测试完成！")
