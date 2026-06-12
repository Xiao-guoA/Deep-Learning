"""
Vision Transformer (ViT) 表情识别模型

使用 Hugging Face 预训练的 ViT 模型
(abhilash88/face-emotion-detection, FER2013 上 71.55% 准确率)

用法:
    from models.vit_expression import ViTExpression
    model = ViTExpression()
    logits = model(images_tensor)  # 兼容 torch.nn.Module 接口
"""

import torch
import torch.nn as nn
from PIL import Image
import numpy as np

# Module-level cache for the image processor (avoids re-loading on every call)
_VIT_PROCESSOR = None


class ViTExpression(nn.Module):
    """
    Hugging Face ViT 表情识别模型封装

    使用预训练权重，开箱即用，无需训练。
    """

    def __init__(self, model_name='abhilash88/face-emotion-detection'):
        super().__init__()

        from transformers import ViTForImageClassification
        self.model = ViTForImageClassification.from_pretrained(model_name)
        self.model.eval()

        # FER2013 标准 7 类表情映射
        self.label_map = {
            0: 'Angry',     # LABEL_0
            1: 'Disgust',   # LABEL_1
            2: 'Fear',      # LABEL_2
            3: 'Happy',     # LABEL_3
            4: 'Sad',       # LABEL_4
            5: 'Surprise',  # LABEL_5
            6: 'Neutral',   # LABEL_6
        }

    @property
    def device(self):
        return next(self.model.parameters()).device

    def to(self, device):
        self.model = self.model.to(device)
        return super().to(device)

    def forward(self, pixel_values):
        """
        前向传播

        参数:
            pixel_values: torch.Tensor, shape=(B, 3, 224, 224)
                          (ImageNet 标准化后的张量)

        返回:
            logits: torch.Tensor, shape=(B, 7)
        """
        outputs = self.model(pixel_values=pixel_values)
        return outputs.logits

    def get_label_name(self, idx):
        """将模型输出索引映射为表情名称"""
        return self.label_map.get(idx, f'Unknown({idx})')


def load_vit_model(device='cpu'):
    """加载 ViT 模型"""
    model = ViTExpression()
    model = model.to(device)
    model.eval()
    return model


def vit_preprocess(image, device=None):
    """
    ViT 图片预处理（processor 全局缓存，避免重复加载）

    参数:
        image: PIL Image
        device: 'cuda' / 'cpu'，指定后张量直接创建在目标设备上

    返回:
        torch.Tensor, shape=(1, 3, 224, 224)
    """
    global _VIT_PROCESSOR
    if _VIT_PROCESSOR is None:
        from transformers import ViTImageProcessor
        _VIT_PROCESSOR = ViTImageProcessor.from_pretrained(
            'abhilash88/face-emotion-detection'
        )
    inputs = _VIT_PROCESSOR(images=image, return_tensors='pt')
    pixel_values = inputs['pixel_values']
    if device is not None:
        pixel_values = pixel_values.to(device)
    return pixel_values


if __name__ == '__main__':
    # 测试
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_vit_model(device)
    dummy = torch.randn(1, 3, 224, 224).to(device)
    out = model(dummy)
    print(f'输入: (1, 3, 224, 224) → 输出: {out.shape}')
    print(f'预测类别: {model.get_label_name(out.argmax().item())}')
    print(f'参数量: {sum(p.numel() for p in model.parameters()):,}')
