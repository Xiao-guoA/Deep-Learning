"""
FER2013 数据集工具

提供 Dataset 类、数据加载和标签映射功能。
"""

import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

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
        transform: torchvision 数据变换（用于训练增强 + 归一化）
    """

    def __init__(self, csv_path, split='Training', transform=None):
        df = pd.read_csv(csv_path)
        self.df = df[df['Usage'] == split].reset_index(drop=True)
        self.transform = transform

        print(f'[FER2013] {split}: {len(self.df)} 个样本')

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # 解析像素 -> 48x48 灰度图 -> 转 3 通道 RGB (ViT 需要)
        img_array = parse_pixels(row['pixels'])
        img = Image.fromarray(img_array, mode='L').convert('RGB')

        label = int(row['emotion'])

        if self.transform:
            img = self.transform(img)

        return img, label


def get_fer2013_dataloaders(csv_path, batch_size=64, num_workers=0):
    """
    创建 FER2013 训练/验证/测试 DataLoader

    参数:
        csv_path: fer2013.csv 路径
        batch_size: 批次大小
        num_workers: 数据加载线程数（Windows 建议设为 0）

    返回:
        train_loader, val_loader, test_loader
    """
    from torchvision import transforms

    # ViT 处理器（用于归一化）
    from transformers import ViTImageProcessor
    processor = ViTImageProcessor.from_pretrained('abhilash88/face-emotion-detection')

    # 获取 ViT 使用的 mean/std
    image_mean = processor.image_mean
    image_std = processor.image_std
    size = processor.size.get('height', 224)

    # 训练集：数据增强 + 归一化
    train_transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.RandomRotation(degrees=10),
        transforms.ToTensor(),
        transforms.Normalize(mean=image_mean, std=image_std),
    ])

    # 验证/测试集：仅缩放 + 归一化
    eval_transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=image_mean, std=image_std),
    ])

    train_set = FER2013Dataset(csv_path, split='Training', transform=train_transform)
    val_set = FER2013Dataset(csv_path, split='PublicTest', transform=eval_transform)
    test_set = FER2013Dataset(csv_path, split='PrivateTest', transform=eval_transform)

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
