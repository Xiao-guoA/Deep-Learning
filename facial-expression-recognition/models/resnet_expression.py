"""
ResNet18 迁移学习模型

使用 torchvision 提供的 ResNet18 预训练权重（ImageNet），
替换最后一层全连接以适应 FER2013 的 7 类表情分类。

支持两种微调策略：
  1. finetune_all=True  — 全模型微调（解冻所有层）
  2. finetune_all=False — 只训练分类头（冻结主干，快速）

输入尺寸: 224x224 RGB（ResNet 标准输入）
"""

import torch
import torch.nn as nn
from torchvision import models


class ResNetExpression(nn.Module):
    """
    ResNet18 + 自定义分类头（7 类表情）

    参数:
        num_classes: 输出类别数（默认 7）
        finetune_all: True=全模型微调, False=只训练分类头
    """

    def __init__(self, num_classes=7, finetune_all=True):
        super().__init__()

        # 加载预训练 ResNet18
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        self.backbone = models.resnet18(weights=weights)

        # 获取 backbone 的特征维度
        in_features = self.backbone.fc.in_features  # 512

        # 替换分类头
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

        # 冻结/解冻策略
        self.finetune_all = finetune_all
        self._set_grad()

    def _set_grad(self):
        """根据 finetune_all 设置 requires_grad"""
        if not self.finetune_all:
            # 冻结 backbone 所有层
            for param in self.backbone.parameters():
                param.requires_grad = False
            # 只训练新分类头
            for param in self.backbone.fc.parameters():
                param.requires_grad = True

    def forward(self, x):
        return self.backbone(x)


def count_parameters(model):
    """统计可训练参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    model = ResNetExpression(finetune_all=True)
    x = torch.randn(1, 3, 224, 224)
    y = model(x)
    total = sum(p.numel() for p in model.parameters())
    trainable = count_parameters(model)
    print(f'输入形状: {x.shape}')
    print(f'输出形状: {y.shape}')
    print(f'总参数量: {total:,}')
    print(f'可训练参数量: {trainable:,}')
    assert y.shape == (1, 7), f'期望 (1, 7), 得到 {y.shape}'
    print('ResNetExpression 前向测试通过 ✓')

    # 测试冻结模式
    print('\n--- 冻结模式 ---')
    model_frozen = ResNetExpression(finetune_all=False)
    trainable_frozen = count_parameters(model_frozen)
    print(f'可训练参数量（冻结）: {trainable_frozen:,}')
