"""
损失函数实现
最简版本，只包含核心功能
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    """
    Dice损失函数
    用于分割任务，处理类别不平衡问题
    """
    
    def __init__(self, smooth=1e-5):
        """
        初始化Dice损失
        
        Args:
            smooth (float): 平滑参数，避免除零
        """
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        """
        计算Dice损失
        
        Args:
            pred (torch.Tensor): 预测结果 (B, C, H, W)
            target (torch.Tensor): 真实标签 (B, H, W)
            
        Returns:
            torch.Tensor: Dice损失值
        """
        # 将预测结果转换为概率
        pred = F.softmax(pred, dim=1)
        
        # 将target转换为one-hot编码
        num_classes = pred.shape[1]
        target_one_hot = F.one_hot(target.long(), num_classes=num_classes).permute(0, 3, 1, 2).float()
        
        # 计算每个类别的Dice系数
        dice_scores = []
        for c in range(num_classes):
            pred_c = pred[:, c]
            target_c = target_one_hot[:, c]
            
            intersection = (pred_c * target_c).sum()
            union = pred_c.sum() + target_c.sum()
            
            dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
            dice_scores.append(dice)
        
        # 返回平均Dice损失
        return 1.0 - torch.stack(dice_scores).mean()

class CombinedLoss(nn.Module):
    """
    组合损失函数
    Cross Entropy + Dice Loss
    """
    
    def __init__(self, ce_weight=1.0, dice_weight=1.0):
        """
        初始化组合损失
        
        Args:
            ce_weight (float): Cross Entropy权重
            dice_weight (float): Dice Loss权重
        """
        super(CombinedLoss, self).__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        
        self.ce_loss = nn.CrossEntropyLoss()
        self.dice_loss = DiceLoss()
    
    def forward(self, pred, target):
        """
        计算组合损失
        
        Args:
            pred (torch.Tensor): 预测结果 (B, C, H, W)
            target (torch.Tensor): 真实标签 (B, H, W)
            
        Returns:
            torch.Tensor: 组合损失值
        """
        ce = self.ce_loss(pred, target.long())
        dice = self.dice_loss(pred, target)
        
        return self.ce_weight * ce + self.dice_weight * dice

def test_loss():
    """测试损失函数"""
    print("=== 测试损失函数 ===")
    
    # 创建测试数据
    batch_size = 2
    num_classes = 5
    height, width = 64, 64
    
    pred = torch.randn(batch_size, num_classes, height, width)
    target = torch.randint(0, num_classes, (batch_size, height, width))
    
    print(f"预测形状: {pred.shape}")
    print(f"目标形状: {target.shape}")
    
    # 测试Cross Entropy损失
    ce_loss = nn.CrossEntropyLoss()
    ce = ce_loss(pred, target.long())
    print(f"Cross Entropy损失: {ce.item():.4f}")
    
    # 测试Dice损失
    dice_loss = DiceLoss()
    dice = dice_loss(pred, target)
    print(f"Dice损失: {dice.item():.4f}")
    
    # 测试组合损失
    combined_loss = CombinedLoss()
    combined = combined_loss(pred, target)
    print(f"组合损失: {combined.item():.4f}")
    
    print("✅ 损失函数测试完成！")

if __name__ == "__main__":
    test_loss()
