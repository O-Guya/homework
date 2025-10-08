"""
测试完整模型
最简版本，只包含核心功能
"""

import torch
from unet import UNet
from loss import CombinedLoss
from optimizer import create_optimizer, create_scheduler

def test_complete_model():
    """测试完整模型"""
    print("=== 测试完整模型 ===")
    
    # 1. 创建模型
    model = UNet(num_classes=5)
    print(f"✅ 模型创建成功，参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 2. 创建损失函数
    criterion = CombinedLoss(ce_weight=1.0, dice_weight=1.0)
    print("✅ 损失函数创建成功")
    
    # 3. 创建优化器
    optimizer = create_optimizer(model, lr=1e-4)
    scheduler = create_scheduler(optimizer, step_size=10, gamma=0.5)
    print("✅ 优化器创建成功")
    
    # 4. 测试前向传播
    batch_size = 2
    x = torch.randn(batch_size, 3, 512, 512)
    target = torch.randint(0, 5, (batch_size, 512, 512))
    
    print(f"输入形状: {x.shape}")
    print(f"目标形状: {target.shape}")
    
    # 前向传播
    model.eval()
    with torch.no_grad():
        output = model(x)
        print(f"输出形状: {output.shape}")
    
    # 5. 测试训练步骤
    model.train()
    optimizer.zero_grad()
    
    output = model(x)
    loss = criterion(output, target)
    
    print(f"损失值: {loss.item():.4f}")
    
    # 反向传播
    loss.backward()
    optimizer.step()
    scheduler.step()
    
    print(f"更新后学习率: {optimizer.param_groups[0]['lr']:.6f}")
    
    print("✅ 完整模型测试成功！")

if __name__ == "__main__":
    test_complete_model()
