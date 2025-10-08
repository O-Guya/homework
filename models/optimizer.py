"""
优化器实现
最简版本，只包含核心功能
"""

import torch
import torch.optim as optim

def create_optimizer(model, lr=1e-4, weight_decay=1e-5):
    """
    创建优化器
    
    Args:
        model (nn.Module): 模型
        lr (float): 学习率
        weight_decay (float): 权重衰减
        
    Returns:
        torch.optim.Optimizer: 优化器
    """
    optimizer = optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )
    return optimizer

def create_scheduler(optimizer, step_size=30, gamma=0.1):
    """
    创建学习率调度器
    
    Args:
        optimizer: 优化器
        step_size (int): 学习率衰减步长
        gamma (float): 学习率衰减因子
        
    Returns:
        torch.optim.lr_scheduler: 学习率调度器
    """
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=step_size,
        gamma=gamma
    )
    return scheduler

def test_optimizer():
    """测试优化器"""
    print("=== 测试优化器 ===")
    
    # 创建简单模型
    model = torch.nn.Linear(10, 1)
    
    # 创建优化器
    optimizer = create_optimizer(model, lr=1e-3)
    scheduler = create_scheduler(optimizer, step_size=10, gamma=0.5)
    
    print(f"初始学习率: {optimizer.param_groups[0]['lr']:.6f}")
    
    # 模拟训练步骤
    for epoch in range(15):
        # 模拟损失
        loss = torch.tensor(1.0, requires_grad=True)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 更新学习率
        scheduler.step()
        
        if epoch % 5 == 0:
            print(f"Epoch {epoch}: 学习率 = {optimizer.param_groups[0]['lr']:.6f}")
    
    print("✅ 优化器测试完成！")

if __name__ == "__main__":
    test_optimizer()
