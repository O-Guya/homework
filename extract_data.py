"""
解压数据集
"""
import zipfile
import os
from pathlib import Path

def extract_dataset():
    """解压数据集"""
    data_dir = "/mnt/external_4tb/hjy/assignment1/data"
    zip_file = data_dir / "ToothSegmDataset.zip"
    
    print("=== 解压数据集 ===")
    
    if zip_file.exists():
        print(f"找到ZIP文件: {zip_file}")
        print(f"ZIP文件大小: {zip_file.stat().st_size / (1024**3):.2f} GB")
        
        # 检查是否已经解压
        extract_dir = data_dir / "ToothSegmDataset"
        if extract_dir.exists():
            print(f"解压目录已存在: {extract_dir}")
            
            # 检查是否有内容
            subdirs = [d for d in extract_dir.iterdir() if d.is_dir()]
            print(f"解压目录中的子目录数量: {len(subdirs)}")
            
            if len(subdirs) > 0:
                print("数据集似乎已经解压，跳过解压过程")
                return
            else:
                print("解压目录存在但为空，重新解压...")
        else:
            print("创建解压目录...")
            extract_dir.mkdir(parents=True, exist_ok=True)
        
        # 解压文件
        print("开始解压...")
        try:
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                # 获取文件列表
                file_list = zip_ref.namelist()
                print(f"ZIP文件中的文件数量: {len(file_list)}")
                
                # 解压所有文件
                zip_ref.extractall(extract_dir)
                print("解压完成！")
                
                # 验证解压结果
                if extract_dir.exists():
                    subdirs = [d for d in extract_dir.iterdir() if d.is_dir()]
                    print(f"解压后子目录数量: {len(subdirs)}")
                    print(f"子目录: {[d.name for d in subdirs[:10]]}")
                
        except Exception as e:
            print(f"解压失败: {e}")
    else:
        print(f"未找到ZIP文件: {zip_file}")
        print("请确保ToothSegmDataset.zip文件存在于data目录中")

if __name__ == "__main__":
    extract_dataset()
