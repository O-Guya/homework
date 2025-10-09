"""
U-Net模型架构，支持牙齿ID条件化 - 修复版本
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import MODEL_CONFIG


class DoubleConv(nn.Module):
    """双卷积块"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """下采样块"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """上采样块 - 修复版本"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 上采样
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # 注意：concatenate后的通道数是 in_channels + out_channels
        # 所以DoubleConv的输入通道数应该是 in_channels + out_channels
        self.conv = DoubleConv(in_channels + out_channels, out_channels)

    def forward(self, x1, x2):
        # x1: 来自更深层的特征 (需要上采样)
        # x2: 来自编码器的特征 (用于skip connection)
        x1 = self.up(x1)
        
        # 处理尺寸不匹配
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        # Concatenate
        x = torch.cat([x2, x1], dim=1)
        
        return self.conv(x)


class OutConv(nn.Module):
    """输出卷积层"""
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class FiLM(nn.Module):
    """Feature-wise Linear Modulation (FiLM) 层"""
    def __init__(self, feature_dim, condition_dim):
        super().__init__()
        self.feature_dim = feature_dim
        self.condition_dim = condition_dim
        
        self.scale_net = nn.Sequential(
            nn.Linear(condition_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)
        )
        self.shift_net = nn.Sequential(
            nn.Linear(condition_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)
        )

    def forward(self, x, condition):
        B, C, H, W = x.shape
        scale = self.scale_net(condition).view(B, C, 1, 1)
        shift = self.shift_net(condition).view(B, C, 1, 1)
        return scale * x + shift


class ToothIDEmbedding(nn.Module):
    """牙齿ID嵌入层"""
    def __init__(self, num_tooth_ids, embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(num_tooth_ids, embedding_dim)
        self.projection = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )

    def forward(self, tooth_ids):
        embedded = self.embedding(tooth_ids)
        embedded = self.projection(embedded)
        return embedded


class ConditionalUNet(nn.Module):
    """条件化U-Net模型"""
    def __init__(self, 
                 input_channels=3, 
                 num_classes=1, 
                 base_filters=64, 
                 num_tooth_ids=31,
                 tooth_id_embedding_dim=32,
                 dropout_rate=0.1):
        super(ConditionalUNet, self).__init__()
        
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.base_filters = base_filters
        
        # 牙齿ID嵌入
        self.tooth_id_embedding = ToothIDEmbedding(num_tooth_ids, tooth_id_embedding_dim)
        
        # 编码器
        self.inc = DoubleConv(input_channels, base_filters)
        self.down1 = Down(base_filters, base_filters * 2)
        self.down2 = Down(base_filters * 2, base_filters * 4)
        self.down3 = Down(base_filters * 4, base_filters * 8)
        self.down4 = Down(base_filters * 8, base_filters * 16)
        
        # FiLM层
        self.film1 = FiLM(base_filters * 2, tooth_id_embedding_dim)
        self.film2 = FiLM(base_filters * 4, tooth_id_embedding_dim)
        self.film3 = FiLM(base_filters * 8, tooth_id_embedding_dim)
        self.film4 = FiLM(base_filters * 16, tooth_id_embedding_dim)
        
        # 解码器
        self.up1 = Up(base_filters * 16, base_filters * 8)
        self.up2 = Up(base_filters * 8, base_filters * 4)
        self.up3 = Up(base_filters * 4, base_filters * 2)
        self.up4 = Up(base_filters * 2, base_filters)
        
        # 输出层
        self.outc = OutConv(base_filters, num_classes)
        
        # Dropout
        self.dropout = nn.Dropout2d(dropout_rate)

    def forward(self, x, tooth_ids):
        # 嵌入牙齿ID
        tooth_embedding = self.tooth_id_embedding(tooth_ids)
        
        # 编码器
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # 应用FiLM条件化
        x2_cond = self.film1(x2, tooth_embedding)
        x3_cond = self.film2(x3, tooth_embedding)
        x4_cond = self.film3(x4, tooth_embedding)
        x5_cond = self.film4(x5, tooth_embedding)
        
        # 解码器
        x = self.up1(x5_cond, x4_cond)
        x = self.dropout(x)
        x = self.up2(x, x3_cond)
        x = self.dropout(x)
        x = self.up3(x, x2_cond)
        x = self.up4(x, x1)
        
        # 输出
        output = self.outc(x)
        
        return output

    def inference(self, input_image, tooth_id):
        """推理接口"""
        self.eval()
        
        with torch.no_grad():
            if input_image.dim() == 3:
                input_image = input_image.unsqueeze(0)
            
            if isinstance(tooth_id, int):
                tooth_id = torch.tensor([tooth_id], dtype=torch.long, device=input_image.device)
            elif tooth_id.dim() == 0:
                tooth_id = tooth_id.unsqueeze(0)
            
            output = self.forward(input_image, tooth_id)
            mask_image = torch.sigmoid(output)
            
        return mask_image


def create_model(config=None):
    """创建模型实例"""
    if config is None:
        config = MODEL_CONFIG
    
    model = ConditionalUNet(
        input_channels=config['input_channels'],
        num_classes=config['num_classes'],
        base_filters=config['base_filters'],
        num_tooth_ids=config['num_tooth_ids'],
        tooth_id_embedding_dim=config['tooth_id_embedding_dim'],
        dropout_rate=config['dropout_rate']
    )
    
    return model


if __name__ == "__main__":
    # 测试模型
    model = create_model()
    
    batch_size = 2
    input_image = torch.randn(batch_size, 3, 256, 256)
    tooth_ids = torch.tensor([0, 1])
    
    output = model(input_image, tooth_ids)
    print(f"输入形状: {input_image.shape}")
    print(f"输出形状: {output.shape}")
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")

