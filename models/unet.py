"""
U-Net模型实现
最简版本，只包含核心功能
"""

import torch
import torch.nn as nn

class UNet(nn.Module):
    """
    U-Net分割模型
    输入: (batch_size, 3, 512, 512)
    输出: (batch_size, num_classes, 512, 512)
    """
    
    def __init__(self, num_classes=5):
        """
        初始化U-Net
        
        Args:
            num_classes (int): 分割类别数，默认5（背景+4个牙齿类别）
        """
        super(UNet, self).__init__()
        self.num_classes = num_classes
        
        # 编码器（下采样）
        self.enc1 = self._conv_block(3, 64)
        self.enc2 = self._conv_block(64, 128)
        self.enc3 = self._conv_block(128, 256)
        self.enc4 = self._conv_block(256, 512)
        
        # 瓶颈层
        self.bottleneck = self._conv_block(512, 1024)
        
        # 解码器（上采样）
        self.dec4 = self._conv_block(512 + 512, 512)
        self.dec3 = self._conv_block(256 + 256, 256)
        self.dec2 = self._conv_block(128 + 128, 128)
        self.dec1 = self._conv_block(64 + 64, 64)
        
        # 最终分类层
        self.final = nn.Conv2d(64, num_classes, kernel_size=1)
        
        # 池化和上采样
        self.pool = nn.MaxPool2d(2)
        self.up = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
    
    def _conv_block(self, in_channels, out_channels):
        """
        卷积块：Conv2d + BatchNorm + ReLU
        
        Args:
            in_channels (int): 输入通道数
            out_channels (int): 输出通道数
            
        Returns:
            nn.Sequential: 卷积块
        """
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x (torch.Tensor): 输入图像 (batch_size, 3, 512, 512)
            
        Returns:
            torch.Tensor: 分割结果 (batch_size, num_classes, 512, 512)
        """
        # 编码器
        e1 = self.enc1(x)        # (B, 64, 512, 512)
        e2 = self.enc2(self.pool(e1))  # (B, 128, 256, 256)
        e3 = self.enc3(self.pool(e2))  # (B, 256, 128, 128)
        e4 = self.enc4(self.pool(e3))  # (B, 512, 64, 64)
        
        # 瓶颈层
        b = self.bottleneck(self.pool(e4))  # (B, 1024, 32, 32)
        
        # 解码器
        d4 = self.up(b)  # (B, 512, 64, 64)
        d4 = torch.cat([d4, e4], dim=1)  # (B, 1024, 64, 64)
        d4 = self.dec4(d4)  # (B, 512, 64, 64)
        
        d3 = self.up3(d4)  # (B, 256, 128, 128)
        d3 = torch.cat([d3, e3], dim=1)  # (B, 512, 128, 128)
        d3 = self.dec3(d3)  # (B, 256, 128, 128)
        
        d2 = self.up2(d3)  # (B, 128, 256, 256)
        d2 = torch.cat([d2, e2], dim=1)  # (B, 256, 256, 256)
        d2 = self.dec2(d2)  # (B, 128, 256, 256)
        
        d1 = self.up1(d2)  # (B, 64, 512, 512)
        d1 = torch.cat([d1, e1], dim=1)  # (B, 128, 512, 512)
        d1 = self.dec1(d1)  # (B, 64, 512, 512)
        
        # 最终分类
        out = self.final(d1)  # (B, num_classes, 512, 512)
        
        return out

def test_unet():
    """测试U-Net模型"""
    print("=== 测试U-Net模型 ===")
    
    # 创建模型
    model = UNet(num_classes=5)
    print(f"模型创建成功，类别数: {model.num_classes}")
    
    # 测试前向传播
    batch_size = 2
    x = torch.randn(batch_size, 3, 512, 512)
    print(f"输入形状: {x.shape}")
    
    with torch.no_grad():
        output = model(x)
        print(f"输出形状: {output.shape}")
        print(f"输出值范围: {output.min():.3f} - {output.max():.3f}")
    
    # 计算参数数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"总参数数量: {total_params:,}")
    
    print("✅ U-Net测试完成！")

if __name__ == "__main__":
    test_unet()
