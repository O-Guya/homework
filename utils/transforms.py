from PIL import Image
import numpy as np
import random

class Resize:
    """
    调整图像尺寸的变换类
    
    功能：
    - 将图像调整到指定尺寸
    - 支持RGB图像和掩码图像
    - 保持宽高比或强制调整
    """
    
    def __init__(self, size, keep_aspect_ratio=True):
        """
        初始化Resize变换
        
        Args:
            size (int or tuple): 目标尺寸，可以是单个数字（正方形）或(width, height)元组
            keep_aspect_ratio (bool): 是否保持宽高比
        """
        self.size = size if isinstance(size, tuple) else (size, size)
        self.keep_aspect_ratio = keep_aspect_ratio
    
    def __call__(self, image):
        """
        对图像应用resize变换
        
        Args:
            image (PIL.Image): 输入图像
            
        Returns:
            PIL.Image: 调整尺寸后的图像
        """
        if self.keep_aspect_ratio:
            # 保持宽高比，使用thumbnail方法
            image.thumbnail(self.size, Image.Resampling.LANCZOS)
            # 创建目标尺寸的空白图像
            new_image = Image.new(image.mode, self.size, (0, 0, 0) if image.mode == 'RGB' else 0)
            # 将调整后的图像粘贴到中心
            new_image.paste(image, ((self.size[0] - image.size[0]) // 2, 
                                  (self.size[1] - image.size[1]) // 2))
            return new_image
        else:
            # 直接调整到目标尺寸
            return image.resize(self.size, Image.Resampling.LANCZOS)

class RandomHorizontalFlip:
    """
    随机水平翻转变换
    
    功能：
    - 随机决定是否水平翻转图像
    - 对RGB图像和掩码图像都适用
    - 用于数据增强，提高模型泛化能力
    """
    
    def __init__(self, probability=0.5):
        """
        初始化随机水平翻转
        
        Args:
            probability (float): 翻转的概率，0-1之间
        """
        self.probability = probability
    
    def __call__(self, image):
        """
        对图像应用随机水平翻转
        
        Args:
            image (PIL.Image): 输入图像
            
        Returns:
            PIL.Image: 可能翻转后的图像
        """
        if random.random() < self.probability:
            return image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        return image

class RandomRotation:
    """
    随机旋转变换
    
    功能：
    - 随机旋转图像一定角度
    - 支持指定旋转角度范围
    - 用于数据增强
    """
    
    def __init__(self, degrees, fill=0):
        """
        初始化随机旋转
        
        Args:
            degrees (float or tuple): 旋转角度范围，可以是单个数字或(min, max)元组
            fill: 旋转后空白区域的填充值
        """
        if isinstance(degrees, (int, float)):
            self.degrees = (-degrees, degrees)
        else:
            self.degrees = degrees
        self.fill = fill
    
    def __call__(self, image):
        """
        对图像应用随机旋转
        
        Args:
            image (PIL.Image): 输入图像
            
        Returns:
            PIL.Image: 旋转后的图像
        """
        angle = random.uniform(self.degrees[0], self.degrees[1])
        return image.rotate(angle, fillcolor=self.fill, resample=Image.Resampling.BILINEAR)

class ToTensor:
    """
    将PIL图像转换为PyTorch张量
    
    功能：
    - 将PIL图像转换为numpy数组，再转换为张量
    - 处理RGB和灰度图像
    - 将像素值从[0,255]范围缩放到[0,1]范围
    """
    
    def __init__(self):
        """初始化ToTensor变换"""
        pass
    
    def __call__(self, image):
        """
        将PIL图像转换为张量
        
        Args:
            image (PIL.Image): 输入PIL图像
            
        Returns:
            numpy.ndarray: 转换后的numpy数组（模拟张量）
        """
        # 将PIL图像转换为numpy数组
        if image.mode == 'RGB':
            # RGB图像：H x W x 3
            array = np.array(image, dtype=np.float32) / 255.0
        elif image.mode == 'L':
            # 灰度图像：H x W
            array = np.array(image, dtype=np.float32) / 255.0
        else:
            raise ValueError(f"不支持的图像模式: {image.mode}")
        
        return array

class Normalize:
    """
    标准化变换
    
    功能：
    - 对图像进行标准化处理
    - 使用指定的均值和标准差
    - 常用于深度学习模型的预处理
    """
    
    def __init__(self, mean, std):
        """
        初始化标准化变换
        
        Args:
            mean (list): 各通道的均值
            std (list): 各通道的标准差
        """
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
    
    def __call__(self, image):
        """
        对图像应用标准化
        
        Args:
            image (numpy.ndarray): 输入图像数组
            
        Returns:
            numpy.ndarray: 标准化后的图像数组
        """
        if len(image.shape) == 3:  # RGB图像
            for i in range(3):
                image[:, :, i] = (image[:, :, i] - self.mean[i]) / self.std[i]
        else:  # 灰度图像
            image = (image - self.mean[0]) / self.std[0]
        
        return image

class Compose:
    """
    组合多个变换
    
    功能：
    - 将多个变换按顺序组合
    - 依次对图像应用每个变换
    - 简化变换的使用
    """
    
    def __init__(self, transforms):
        """
        初始化组合变换
        
        Args:
            transforms (list): 变换列表
        """
        self.transforms = transforms
    
    def __call__(self, image):
        """
        对图像应用所有变换
        
        Args:
            image (PIL.Image): 输入图像
            
        Returns:
            numpy.ndarray: 经过所有变换后的图像数组
        """
        for transform in self.transforms:
            image = transform(image)
        return image

# 预定义的变换组合
def get_train_transforms():
    """
    获取训练时的数据变换
    
    Returns:
        Compose: 训练变换组合
    """
    return Compose([
        Resize((512, 512)),
        RandomHorizontalFlip(0.5),
        RandomRotation(10),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def get_val_transforms():
    """
    获取验证时的数据变换
    
    Returns:
        Compose: 验证变换组合
    """
    return Compose([
        Resize((512, 512)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def get_mask_transforms():
    """
    获取掩码的变换（不需要标准化）
    
    Returns:
        Compose: 掩码变换组合
    """
    return Compose([
        Resize((512, 512)),
        ToTensor()
    ])

# 测试函数
def test_transforms():
    """测试变换功能"""
    print("=== 测试数据变换 ===")
    
    # 创建测试图像
    test_image = Image.new('RGB', (256, 256), (255, 0, 0))  # 红色图像
    test_mask = Image.new('L', (256, 256), 128)  # 灰色掩码
    
    print(f"原始图像尺寸: {test_image.size}")
    print(f"原始掩码尺寸: {test_mask.size}")
    
    # 测试Resize
    resize = Resize((512, 512))
    resized_image = resize(test_image)
    resized_mask = resize(test_mask)
    print(f"Resize后图像尺寸: {resized_image.size}")
    print(f"Resize后掩码尺寸: {resized_mask.size}")
    
    # 测试ToTensor
    to_tensor = ToTensor()
    image_array = to_tensor(resized_image)
    mask_array = to_tensor(resized_mask)
    print(f"图像数组形状: {image_array.shape}")
    print(f"图像数组值范围: {image_array.min():.3f} - {image_array.max():.3f}")
    print(f"掩码数组形状: {mask_array.shape}")
    print(f"掩码数组值范围: {mask_array.min():.3f} - {mask_array.max():.3f}")
    
    # 测试组合变换
    train_transform = get_train_transforms()
    result = train_transform(test_image)
    print(f"训练变换结果形状: {result.shape}")
    print(f"训练变换结果值范围: {result.min():.3f} - {result.max():.3f}")

if __name__ == "__main__":
    test_transforms()
