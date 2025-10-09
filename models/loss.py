"""
损失函数实现
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Dice损失函数
    """
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        """
        Args:
            pred: 预测结果 (B, 1, H, W)
            target: 真实标签 (B, 1, H, W)
        Returns:
            dice_loss: Dice损失
        """
        # 应用sigmoid激活
        pred = torch.sigmoid(pred)
        
        # 展平
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        # 计算交集和并集
        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum()
        
        # 计算Dice系数
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        
        # 返回Dice损失 (1 - Dice)
        return 1 - dice


class FocalLoss(nn.Module):
    """
    Focal损失函数，用于处理类别不平衡
    """
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred, target):
        """
        Args:
            pred: 预测结果 (B, 1, H, W)
            target: 真实标签 (B, 1, H, W)
        Returns:
            focal_loss: Focal损失
        """
        # 计算BCE损失
        bce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        
        # 计算概率
        pt = torch.exp(-bce_loss)
        
        # 计算Focal损失
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class TverskyLoss(nn.Module):
    """
    Tversky损失函数，Dice损失的泛化版本
    """
    def __init__(self, alpha=0.3, beta=0.7, smooth=1e-6):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha  # 假阳性权重
        self.beta = beta    # 假阴性权重
        self.smooth = smooth

    def forward(self, pred, target):
        """
        Args:
            pred: 预测结果 (B, 1, H, W)
            target: 真实标签 (B, 1, H, W)
        Returns:
            tversky_loss: Tversky损失
        """
        # 应用sigmoid激活
        pred = torch.sigmoid(pred)
        
        # 展平
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        # 计算真阳性、假阳性、假阴性
        true_pos = (pred_flat * target_flat).sum()
        false_pos = (pred_flat * (1 - target_flat)).sum()
        false_neg = ((1 - pred_flat) * target_flat).sum()
        
        # 计算Tversky系数
        tversky = (true_pos + self.smooth) / (
            true_pos + self.alpha * false_pos + self.beta * false_neg + self.smooth
        )
        
        # 返回Tversky损失 (1 - Tversky)
        return 1 - tversky


class CombinedLoss(nn.Module):
    """
    组合损失函数
    结合BCE损失和Dice损失
    """
    def __init__(self, bce_weight=1.0, dice_weight=1.0, focal_weight=0.0, 
                 focal_alpha=1.0, focal_gamma=2.0):
        super(CombinedLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss()
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)

    def forward(self, pred, target):
        """
        Args:
            pred: 预测结果 (B, 1, H, W)
            target: 真实标签 (B, 1, H, W)
        Returns:
            combined_loss: 组合损失
        """
        # 计算各种损失
        bce = self.bce_loss(pred, target)
        dice = self.dice_loss(pred, target)
        
        # 组合损失
        loss = self.bce_weight * bce + self.dice_weight * dice
        
        # 可选添加Focal损失
        if self.focal_weight > 0:
            focal = self.focal_loss(pred, target)
            loss += self.focal_weight * focal
        
        return loss


class IoULoss(nn.Module):
    """
    IoU损失函数
    """
    def __init__(self, smooth=1e-6):
        super(IoULoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        """
        Args:
            pred: 预测结果 (B, 1, H, W)
            target: 真实标签 (B, 1, H, W)
        Returns:
            iou_loss: IoU损失
        """
        # 应用sigmoid激活
        pred = torch.sigmoid(pred)
        
        # 展平
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        # 计算交集和并集
        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum() - intersection
        
        # 计算IoU
        iou = (intersection + self.smooth) / (union + self.smooth)
        
        # 返回IoU损失 (1 - IoU)
        return 1 - iou


def create_loss_function(loss_type='combined', **kwargs):
    """
    创建损失函数
    Args:
        loss_type: 损失函数类型
        **kwargs: 损失函数参数
    Returns:
        loss_fn: 损失函数
    """
    if loss_type == 'bce':
        return nn.BCEWithLogitsLoss()
    elif loss_type == 'dice':
        return DiceLoss(**kwargs)
    elif loss_type == 'focal':
        return FocalLoss(**kwargs)
    elif loss_type == 'tversky':
        return TverskyLoss(**kwargs)
    elif loss_type == 'iou':
        return IoULoss(**kwargs)
    elif loss_type == 'combined':
        return CombinedLoss(**kwargs)
    else:
        raise ValueError(f"未知的损失函数类型: {loss_type}")


if __name__ == "__main__":
    # 测试损失函数
    pred = torch.randn(2, 1, 64, 64)
    target = torch.randint(0, 2, (2, 1, 64, 64)).float()
    
    # 测试各种损失函数
    losses = {
        'BCE': nn.BCEWithLogitsLoss(),
        'Dice': DiceLoss(),
        'Focal': FocalLoss(),
        'Tversky': TverskyLoss(),
        'IoU': IoULoss(),
        'Combined': CombinedLoss()
    }
    
    print("损失函数测试:")
    for name, loss_fn in losses.items():
        loss_value = loss_fn(pred, target)
        print(f"{name}: {loss_value.item():.4f}")
