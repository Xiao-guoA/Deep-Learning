"""
自建 CNN 基线模型 —— 从零训练，不依赖预训练权重

架构设计思路：
  - 深度适中的卷积栈（3 个 Conv-BN-ReLU block），逐步增加通道数
  - 每个 block 后跟 MaxPool 降采样，减小特征图尺寸
  - 使用 BatchNorm 加速收敛 + Dropout 防过拟合
  - 最后接全局平均池化 + 全连接分类头

输入:  48x48 RGB 图像（FER2013 原始分辨率）
输出:  7 类表情 logits
参数量: ~55 万（远小于 ViT 的 8580 万，4GB 显卡可轻松训练）
"""

import torch
import torch.nn as nn


class CNNBaseline(nn.Module):
    """
    CNN 基线模型

    结构:
      Conv1(3→64) → BN → ReLU
      Conv2(64→64) → BN → ReLU → MaxPool(2)
      Conv3(64→128) → BN → ReLU
      Conv4(128→128) → BN → ReLU → MaxPool(2)
      Conv5(128→256) → BN → ReLU → MaxPool(2)
      AdaptiveAvgPool → Dropout → FC(256→7)

    输入尺寸: 48x48
    Feature map 变化:
      48x48 → 48x48 → 24x24 → 24x24 → 12x12 → 6x6 → 1x1
    """

    def __init__(self, num_classes=7, dropout=0.5):
        super().__init__()

        # Block 1: 3→64, 48x48→48x48
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # Block 2: 64→128, 24x24→24x24
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        # Block 3: 128→256, 12x12→12x12
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        # 池化层（单独定义，方便 forward 中灵活使用）
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # 分类头
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),   # 256x1x1
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

        # 参数初始化
        self._init_weights()

    def _init_weights(self):
        """Kaiming 初始化（适配 ReLU）"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # Block 1 + pool: 48x48 → 24x24
        x = self.conv1(x)
        x = self.pool(x)

        # Block 2 + pool: 24x24 → 12x12
        x = self.conv2(x)
        x = self.pool(x)

        # Block 3 + pool: 12x12 → 6x6
        x = self.conv3(x)
        x = self.pool(x)

        # 分类头: 6x6 → 1x1 → 7
        x = self.classifier(x)
        return x


def count_parameters(model):
    """统计可训练参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    model = CNNBaseline()
    x = torch.randn(1, 3, 48, 48)
    y = model(x)
    print(f'输入形状: {x.shape}')
    print(f'输出形状: {y.shape}')
    print(f'参数量: {count_parameters(model):,}')
    assert y.shape == (1, 7), f'期望 (1, 7), 得到 {y.shape}'
    print('CNNBaseline 前向测试通过 ✓')
