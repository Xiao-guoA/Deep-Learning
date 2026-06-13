"""
FER2013 数据集工具

提供 Dataset 类和数据加载函数，支持两种模型：
  - cnn:    保持 48x48 原始分辨率（自建 CNN 用）
  - resnet: 缩放到 224x224（ResNet18 迁移学习用）

去掉了 ViTImageProcessor 依赖，全部使用 torchvision.transforms。
"""

import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

# FER2013 标准 7 类表情标签
EMOTION_LABELS = {
    0: 'Angry',
    1: 'Disgust',
    2: 'Fear',
    3: 'Happy',
    4: 'Sad',
    5: 'Surprise',
    6: 'Neutral',
}

EMOTION_LABELS_CN = {
    0: '生气',
    1: '厌恶',
    2: '恐惧',
    3: '开心',
    4: '悲伤',
    5: '惊讶',
    6: '中性',
}

NUM_CLASSES = 7


def parse_pixels(pixel_str):
    """将 CSV 中的像素字符串解析为 48x48 numpy 数组"""
    return np.array([int(p) for p in pixel_str.split()], dtype=np.uint8).reshape(48, 48)


class FER2013Dataset(Dataset):
    """
    FER2013 数据集（从本地 CSV 文件加载）

    参数:
        csv_path: fer2013.csv 文件路径
        split: 'Training' / 'PublicTest' / 'PrivateTest'
        model_type: 'cnn' 或 'resnet'，决定输入尺寸和增强策略
        augment: 是否使用数据增强（仅训练集用）
    """

    def __init__(self, csv_path, split='Training', model_type='cnn', augment=True):
        df = pd.read_csv(csv_path)
        self.df = df[df['Usage'] == split].reset_index(drop=True)
        self.model_type = model_type

        if model_type == 'cnn':
            # CNN：保持 48x48 原始分辨率
            self.input_size = 48
        else:
            # ResNet：缩放到 224x224
            self.input_size = 224

        # 构建 transform
        self.transform = self._build_transform(split, augment)

        print(f'[FER2013] {split}: {len(self.df)} 个样本 (size={self.input_size})')

    def _build_transform(self, split, augment):
        """根据模型类型和数据集划分构建变换流水线"""
        # 通用变换：灰度→RGB + 缩放
        base_transforms = [
            transforms.Resize((self.input_size, self.input_size)),
            transforms.ToTensor(),
        ]

        # CNN 用 ImageNet 标准化（torchvision 预训练模型的标准值）
        # 虽然 CNN 是自训练的，但使用相同标准化有利于收敛
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

        if split == 'Training' and augment:
            # 训练集：数据增强 + 标准化
            return transforms.Compose([
                transforms.Resize((self.input_size, self.input_size)),
                transforms.RandomHorizontalFlip(p=0.3),
                transforms.RandomRotation(degrees=10),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
                transforms.ToTensor(),
                normalize,
            ])
        else:
            # 验证/测试集：仅缩放 + 标准化
            return transforms.Compose(base_transforms + [normalize])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # 解析像素 -> 48x48 灰度图 -> 转 3 通道 RGB
        img_array = parse_pixels(row['pixels'])
        img = Image.fromarray(img_array, mode='L').convert('RGB')

        label = int(row['emotion'])

        if self.transform:
            img = self.transform(img)

        return img, label


def get_fer2013_dataloaders(csv_path, batch_size=64, model_type='cnn',
                            num_workers=0, augment=True):
    """
    创建 FER2013 训练/验证/测试 DataLoader

    参数:
        csv_path:    fer2013.csv 路径
        batch_size:  批次大小
        model_type:  'cnn' 或 'resnet'
        num_workers: 数据加载线程数（Windows 建议 0）
        augment:     是否使用数据增强

    返回:
        train_loader, val_loader, test_loader
    """
    train_set = FER2013Dataset(csv_path, split='Training',
                                model_type=model_type, augment=augment)
    val_set = FER2013Dataset(csv_path, split='PublicTest',
                              model_type=model_type, augment=False)
    test_set = FER2013Dataset(csv_path, split='PrivateTest',
                               model_type=model_type, augment=False)

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, val_loader, test_loader
