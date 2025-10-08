"""
检查文件结构
"""
import os
from pathlib import Path

def check_file_structure():
    """检查文件结构"""
    data_dir = "/mnt/external_4tb/hjy/assignment1/data/ToothSegmDataset"
    
    print("=== 检查文件结构 ===")
    
    # 首先检查是否有zip文件需要解压
    zip_files = list(Path(data_dir).parent.glob('*.zip'))
    print(f"找到ZIP文件: {[f.name for f in zip_files]}")
    
    # 检查trainset_valset
    trainset_dir = Path(data_dir) / 'trainset_valset'
    if trainset_dir.exists():
        print(f"trainset_valset目录存在")
        
        # 检查第一个牙齿类型
        tooth_dirs = [d for d in trainset_dir.iterdir() if d.is_dir()]
        if tooth_dirs:
            sample_tooth = tooth_dirs[0]
            print(f"示例牙齿类型: {sample_tooth.name}")
            
            # 检查train目录
            train_dir = sample_tooth / 'train'
            if train_dir.exists():
                print(f"train目录存在")
                patient_dirs = [d for d in train_dir.iterdir() if d.is_dir()]
                print(f"患者目录数量: {len(patient_dirs)}")
                
                if patient_dirs:
                    sample_patient = patient_dirs[0]
                    print(f"示例患者: {sample_patient.name}")
                    
                    # 列出所有文件
                    all_files = list(sample_patient.iterdir())
                    print(f"患者目录中的文件数量: {len(all_files)}")
                    print(f"前10个文件: {[f.name for f in all_files[:10]]}")
                    
                    # 检查RGB文件
                    rgb_files = list(sample_patient.glob('*_rgb.jpg'))
                    print(f"RGB文件数量: {len(rgb_files)}")
                    if rgb_files:
                        print(f"RGB文件示例: {[f.name for f in rgb_files[:5]]}")
                    
                    # 检查掩码文件
                    mask_files = list(sample_patient.glob('*_mask.jpg'))
                    print(f"掩码文件数量: {len(mask_files)}")
                    if mask_files:
                        print(f"掩码文件示例: {[f.name for f in mask_files[:5]]}")
                    
                    # 检查其他可能的文件格式
                    jpg_files = list(sample_patient.glob('*.jpg'))
                    print(f"所有JPG文件数量: {len(jpg_files)}")
                    if jpg_files:
                        print(f"JPG文件示例: {[f.name for f in jpg_files[:10]]}")
                    
                    # 检查隐藏文件
                    hidden_files = list(sample_patient.glob('.*'))
                    print(f"隐藏文件数量: {len(hidden_files)}")
                    if hidden_files:
                        print(f"隐藏文件: {[f.name for f in hidden_files]}")
            
            # 检查val目录
            val_dir = sample_tooth / 'val'
            if val_dir.exists():
                print(f"val目录存在")
                val_patients = [d for d in val_dir.iterdir() if d.is_dir()]
                print(f"验证集患者数量: {len(val_patients)}")
                
                if val_patients:
                    sample_val_patient = val_patients[0]
                    val_files = list(sample_val_patient.iterdir())
                    print(f"验证集患者文件数量: {len(val_files)}")
                    print(f"验证集文件示例: {[f.name for f in val_files[:5]]}")
    
    # 检查testset
    testset_dir = Path(data_dir) / 'testset'
    if testset_dir.exists():
        print(f"\ntestset目录存在")
        test_dirs = [d for d in testset_dir.iterdir() if d.is_dir()]
        print(f"testset子目录数量: {len(test_dirs)}")
        
        if test_dirs:
            sample_test = test_dirs[0]
            print(f"示例测试目录: {sample_test.name}")
            test_files = list(sample_test.iterdir())
            print(f"测试文件数量: {len(test_files)}")
            if test_files:
                print(f"测试文件示例: {[f.name for f in test_files[:5]]}")
    
    # 检查磁盘空间
    import shutil
    total, used, free = shutil.disk_usage(data_dir)
    print(f"\n磁盘空间:")
    print(f"总空间: {total // (1024**3)} GB")
    print(f"已使用: {used // (1024**3)} GB") 
    print(f"可用空间: {free // (1024**3)} GB")

if __name__ == "__main__":
    check_file_structure()
