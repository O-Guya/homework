"""
测试脚本 - 牙齿分割模型
"""
import os
import argparse
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
from datetime import datetime

from config import *
from models.unet import create_model
from utils.dataset import create_test_loader
from utils.metrics import MetricsTracker
from utils.visualization import show_anns, save_prediction_visualization


class Tester:
    """测试器类"""
    def __init__(self, model, test_loader, device, selected_tooth_ids):
        self.model = model
        self.test_loader = test_loader
        self.device = device
        self.selected_tooth_ids = selected_tooth_ids
        
        # 指标跟踪器
        self.metrics_tracker = MetricsTracker()
        
        # 测试结果
        self.test_results = []
        self.tooth_id_results = {}

    def test_single_tooth_id(self, tooth_id, num_samples=5):
        """
        测试单个牙齿ID
        Args:
            tooth_id: 牙齿ID
            num_samples: 测试样本数量
        Returns:
            results: 测试结果
        """
        print(f"\n测试牙齿ID {tooth_id} (目标样本数: {num_samples})")
        
        # 重置指标跟踪器
        self.metrics_tracker.reset()
        
        # 创建测试数据加载器（只包含指定牙齿ID的数据）
        # 这里我们需要从训练数据中获取测试样本
        from utils.dataset import ToothSegmentationDataset
        
        test_dataset = ToothSegmentationDataset(
            data_path=TRAIN_DATA_PATH,
            tooth_ids=[tooth_id],
            is_training=False
        )
        
        # 限制样本数量
        if len(test_dataset) > num_samples:
            indices = np.random.choice(len(test_dataset), num_samples, replace=False)
            test_dataset = torch.utils.data.Subset(test_dataset, indices)
        
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0
        )
        
        results = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(test_loader, desc=f"测试牙齿ID {tooth_id}")):
                # 获取数据
                image = batch['image'].to(self.device)
                mask = batch['mask'].to(self.device)
                tooth_id_tensor = batch['tooth_id'].to(self.device)
                patient_id = batch['patient_id'][0]
                image_id = batch['image_id'][0]
                
                # 前向传播
                output = self.model(image, tooth_id_tensor)
                pred_mask = torch.sigmoid(output)
                
                # 计算指标
                self.metrics_tracker.update(pred_mask, mask)
                
                # 获取指标
                metrics = self.metrics_tracker.get_average_metrics()
                
                # 保存结果
                result = {
                    'tooth_id': tooth_id,
                    'patient_id': patient_id,
                    'image_id': image_id,
                    'pixel_accuracy': metrics['pixel_accuracy'],
                    'mean_pixel_accuracy': metrics['mean_pixel_accuracy'],
                    'iou': metrics['iou'],
                    'mean_iou': metrics['mean_iou'],
                    'dice': metrics['dice']
                }
                results.append(result)
                
                # 保存可视化结果
                self.save_prediction_visualization(
                    image[0].cpu(),
                    mask[0].cpu(),
                    pred_mask[0].cpu(),
                    tooth_id,
                    patient_id,
                    image_id,
                    batch_idx
                )
        
        # 计算平均指标
        avg_metrics = self.metrics_tracker.get_average_metrics()
        std_metrics = self.metrics_tracker.get_std_metrics()
        
        print(f"牙齿ID {tooth_id} 测试结果:")
        print(f"  像素准确率: {avg_metrics['pixel_accuracy']:.4f} ± {std_metrics['pixel_accuracy']:.4f}")
        print(f"  平均像素准确率: {avg_metrics['mean_pixel_accuracy']:.4f} ± {std_metrics['mean_pixel_accuracy']:.4f}")
        print(f"  IoU: {avg_metrics['iou']:.4f} ± {std_metrics['iou']:.4f}")
        print(f"  平均IoU: {avg_metrics['mean_iou']:.4f} ± {std_metrics['mean_iou']:.4f}")
        print(f"  Dice系数: {avg_metrics['dice']:.4f} ± {std_metrics['dice']:.4f}")
        
        return results, avg_metrics

    def test_all_selected_tooth_ids(self):
        """测试所有选择的牙齿ID"""
        print("开始测试所有选择的牙齿ID...")
        
        all_results = []
        tooth_id_summary = {}
        
        for tooth_id in self.selected_tooth_ids:
            try:
                results, avg_metrics = self.test_single_tooth_id(tooth_id, num_samples=5)
                all_results.extend(results)
                tooth_id_summary[tooth_id] = avg_metrics
            except Exception as e:
                print(f"测试牙齿ID {tooth_id} 时出错: {e}")
                continue
        
        # 保存详细结果
        self.save_detailed_results(all_results)
        
        # 保存汇总结果
        self.save_summary_results(tooth_id_summary)
        
        # 计算总体指标
        self.calculate_overall_metrics(all_results)
        
        return all_results, tooth_id_summary

    def save_prediction_visualization(self, image, true_mask, pred_mask, tooth_id, patient_id, image_id, batch_idx):
        """保存预测可视化结果"""
        # 创建可视化
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        
        # 原始图像
        image_np = image.permute(1, 2, 0).numpy()
        image_np = (image_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]))
        image_np = np.clip(image_np, 0, 1)
        axes[0].imshow(image_np)
        axes[0].set_title(f'原始图像\n牙齿ID: {tooth_id}')
        axes[0].axis('off')
        
        # 真实mask
        true_mask_np = true_mask.squeeze().numpy()
        axes[1].imshow(true_mask_np, cmap='gray')
        axes[1].set_title('真实Mask')
        axes[1].axis('off')
        
        # 预测mask
        pred_mask_np = pred_mask.squeeze().numpy()
        axes[2].imshow(pred_mask_np, cmap='gray')
        axes[2].set_title('预测Mask')
        axes[2].axis('off')
        
        # 叠加显示
        overlay = image_np.copy()
        pred_binary = (pred_mask_np > 0.5).astype(np.uint8)
        overlay[pred_binary > 0] = [1, 0, 0]  # 红色表示预测区域
        axes[3].imshow(overlay)
        axes[3].set_title('叠加显示')
        axes[3].axis('off')
        
        plt.tight_layout()
        
        # 保存图像
        save_path = os.path.join(
            VISUALIZATION_SAVE_PATH,
            f'tooth_{tooth_id}_patient_{patient_id}_image_{image_id}_batch_{batch_idx}.png'
        )
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def save_detailed_results(self, results):
        """保存详细测试结果"""
        df = pd.DataFrame(results)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(RESULT_SAVE_PATH, f'detailed_test_results_{timestamp}.csv')
        df.to_csv(save_path, index=False)
        print(f"详细结果已保存到: {save_path}")

    def save_summary_results(self, tooth_id_summary):
        """保存汇总结果"""
        summary_data = []
        for tooth_id, metrics in tooth_id_summary.items():
            summary_data.append({
                'tooth_id': tooth_id,
                'pixel_accuracy': metrics['pixel_accuracy'],
                'mean_pixel_accuracy': metrics['mean_pixel_accuracy'],
                'iou': metrics['iou'],
                'mean_iou': metrics['mean_iou'],
                'dice': metrics['dice']
            })
        
        df = pd.DataFrame(summary_data)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(RESULT_SAVE_PATH, f'summary_test_results_{timestamp}.csv')
        df.to_csv(save_path, index=False)
        print(f"汇总结果已保存到: {save_path}")
        
        # 打印汇总表格
        print("\n汇总结果:")
        print(df.to_string(index=False, float_format='%.4f'))

    def calculate_overall_metrics(self, results):
        """计算总体指标"""
        if not results:
            print("没有测试结果可计算")
            return
        
        # 计算总体平均指标
        overall_metrics = {
            'pixel_accuracy': np.mean([r['pixel_accuracy'] for r in results]),
            'mean_pixel_accuracy': np.mean([r['mean_pixel_accuracy'] for r in results]),
            'iou': np.mean([r['iou'] for r in results]),
            'mean_iou': np.mean([r['mean_iou'] for r in results]),
            'dice': np.mean([r['dice'] for r in results])
        }
        
        # 计算标准差
        overall_std = {
            'pixel_accuracy': np.std([r['pixel_accuracy'] for r in results]),
            'mean_pixel_accuracy': np.std([r['mean_pixel_accuracy'] for r in results]),
            'iou': np.std([r['iou'] for r in results]),
            'mean_iou': np.std([r['mean_iou'] for r in results]),
            'dice': np.std([r['dice'] for r in results])
        }
        
        print("\n总体测试结果:")
        print(f"  像素准确率: {overall_metrics['pixel_accuracy']:.4f} ± {overall_std['pixel_accuracy']:.4f}")
        print(f"  平均像素准确率: {overall_metrics['mean_pixel_accuracy']:.4f} ± {overall_std['mean_pixel_accuracy']:.4f}")
        print(f"  IoU: {overall_metrics['iou']:.4f} ± {overall_std['iou']:.4f}")
        print(f"  平均IoU: {overall_metrics['mean_iou']:.4f} ± {overall_std['mean_iou']:.4f}")
        print(f"  Dice系数: {overall_metrics['dice']:.4f} ± {overall_std['dice']:.4f}")
        
        # 保存总体结果
        overall_data = {
            'metric': list(overall_metrics.keys()),
            'mean': list(overall_metrics.values()),
            'std': list(overall_std.values())
        }
        
        df_overall = pd.DataFrame(overall_data)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(RESULT_SAVE_PATH, f'overall_test_results_{timestamp}.csv')
        df_overall.to_csv(save_path, index=False)
        print(f"总体结果已保存到: {save_path}")


def inference(input_image, tooth_id, model_path, device='auto'):
    """
    推理函数，符合prompt要求
    Args:
        input_image: 输入图像 (numpy array 或 torch tensor)
        tooth_id: 目标牙齿ID (int)
        model_path: 模型路径
    Returns:
        mask_image: 分割mask (numpy array)
    """
    # 设备选择
    if device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)
    
    # 加载模型
    model = create_model(MODEL_CONFIG)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # 预处理输入图像
    if isinstance(input_image, np.ndarray):
        # 如果是numpy数组，转换为tensor
        if input_image.dtype != np.float32:
            input_image = input_image.astype(np.float32)
        if input_image.max() > 1.0:
            input_image = input_image / 255.0
        
        # 调整维度
        if input_image.shape[2] == 3:  # HWC -> CHW
            input_image = input_image.transpose(2, 0, 1)
        
        input_tensor = torch.from_numpy(input_image).unsqueeze(0)  # 添加batch维度
    else:
        input_tensor = input_image
    
    # 调整图像尺寸
    if input_tensor.shape[-2:] != (256, 256):
        input_tensor = torch.nn.functional.interpolate(
            input_tensor, size=(256, 256), mode='bilinear', align_corners=False
        )
    
    # 归一化
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    input_tensor = (input_tensor - mean) / std
    
    input_tensor = input_tensor.to(device)
    
    # 推理
    with torch.no_grad():
        mask_tensor = model.inference(input_tensor, tooth_id)
    
    # 转换为numpy数组
    mask_image = mask_tensor.squeeze().cpu().numpy()
    
    return mask_image


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='牙齿分割模型测试')
    parser.add_argument('--model_path', type=str, required=True, help='模型路径')
    parser.add_argument('--tooth_ids', nargs='+', type=int, default=TRAIN_CONFIG['selected_tooth_ids'],
                       help='要测试的牙齿ID列表')
    parser.add_argument('--device', type=str, default='auto', help='设备 (cuda/cpu/auto)')
    parser.add_argument('--num_samples', type=int, default=5, help='每个牙齿ID的测试样本数')
    
    args = parser.parse_args()
    
    # 设备选择
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"使用设备: {device}")
    print(f"模型路径: {args.model_path}")
    print(f"测试牙齿ID: {args.tooth_ids}")
    
    # 加载模型
    print("加载模型...")
    model = create_model(MODEL_CONFIG)
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"模型加载完成，训练轮数: {checkpoint.get('epoch', 'unknown')}")
    
    # 创建测试器
    test_loader = create_test_loader()
    tester = Tester(model, test_loader, device, args.tooth_ids)
    
    # 开始测试
    all_results, tooth_id_summary = tester.test_all_selected_tooth_ids()
    
    print("\n测试完成！")
    print(f"可视化结果保存在: {VISUALIZATION_SAVE_PATH}")
    print(f"测试结果保存在: {RESULT_SAVE_PATH}")


if __name__ == "__main__":
    main()
