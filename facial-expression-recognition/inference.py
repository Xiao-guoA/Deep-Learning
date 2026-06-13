"""
人脸表情识别推理脚本

支持自建 CNN 和 ResNet18 两种模型，支持单张和批量图片推理。

用法:
    # 自建 CNN（默认）
    python inference.py --image 图片路径.jpg

    # ResNet18
    python inference.py --image 图片路径.jpg --model resnet18
"""

import os
import sys
import argparse
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 标签映射
EMOTION_LABELS = {
    0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy',
    4: 'Sad', 5: 'Surprise', 6: 'Neutral',
}

# 标准归一化参数
NORMALIZE = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
)


def get_transform(model_type):
    """根据模型类型获取预处理 transform"""
    if model_type == 'cnn':
        size = 48
    else:
        size = 224
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        NORMALIZE,
    ])


def load_model(model_type='cnn', checkpoint_path=None, device='cuda'):
    """加载模型权重"""
    if model_type == 'cnn':
        from models.cnn_baseline import CNNBaseline
        model = CNNBaseline(num_classes=7)
        model_name = '自建 CNN'
    elif model_type == 'resnet18':
        from models.resnet_expression import ResNetExpression
        model = ResNetExpression(num_classes=7, finetune_all=True)
        model_name = 'ResNet18'
    else:
        raise ValueError(f'未知模型: {model_type}')

    # 加载权重
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f'已加载权重: {checkpoint_path}')
    else:
        print(f'警告: 未找到权重文件 {checkpoint_path}，使用随机初始化')

    model = model.to(device)
    model.eval()
    params = sum(p.numel() for p in model.parameters())
    print(f'{model_name} 加载完成 ({params:,} 参数)')
    return model


@torch.no_grad()
def predict(model, image_tensor, device='cuda'):
    """
    预测单张图片的表情

    返回:
        emotion: 预测的表情名称
        confidence: 置信度
        probs: 各类别概率字典
    """
    image_tensor = image_tensor.to(device)
    outputs = model(image_tensor)
    probabilities = torch.softmax(outputs, dim=1)

    confidence, predicted = torch.max(probabilities, dim=1)
    confidence = confidence.item()
    predicted = predicted.item()

    emotion = EMOTION_LABELS[predicted]
    probs = probabilities[0].cpu().numpy()
    probs_dict = {EMOTION_LABELS[i]: float(probs[i]) for i in range(7)}

    return emotion, confidence, probs_dict


def predict_image(model, image_path, model_type='cnn', device='cuda'):
    """预测图片文件的表情"""
    image = Image.open(image_path).convert('RGB')
    transform = get_transform(model_type)
    tensor = transform(image).unsqueeze(0).to(device)
    return predict(model, tensor, device)


def main():
    parser = argparse.ArgumentParser(description='人脸表情识别推理')
    parser.add_argument('--image', type=str, help='输入图片路径')
    parser.add_argument('--image-dir', type=str, help='批量处理目录下的图片')
    parser.add_argument('--model', type=str, default='cnn',
                        choices=['cnn', 'resnet18'],
                        help='模型类型: cnn (自建) / resnet18 (迁移学习)')
    parser.add_argument('--checkpoint', type=str,
                        default='checkpoints/cnn/best_model.pth',
                        help='模型权重路径')
    parser.add_argument('--cpu', action='store_true', help='强制使用 CPU')

    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
    print(f'设备: {device}')
    print(f'模型: {args.model}')

    model = load_model(args.model, args.checkpoint, device)

    if args.image:
        emotion, confidence, probs = predict_image(
            model, args.image, args.model, device
        )
        print(f'\n图片: {args.image}')
        print(f'预测结果: {emotion} (置信度: {confidence*100:.1f}%)')
        print('\n各类别概率:')
        for emo, prob in sorted(probs.items(), key=lambda x: x[1], reverse=True):
            bar = '#' * int(prob * 50)
            print(f'  {emo:10s}: {prob*100:5.1f}% {bar}')

    if args.image_dir:
        print(f'\n批量处理: {args.image_dir}')
        extensions = ('.jpg', '.jpeg', '.png', '.bmp')
        files = [f for f in os.listdir(args.image_dir) if f.lower().endswith(extensions)]
        print(f'找到 {len(files)} 张图片')

        for fname in sorted(files):
            path = os.path.join(args.image_dir, fname)
            emotion, confidence, _ = predict_image(model, path, args.model, device)
            print(f'  {fname:30s} -> {emotion:8s} ({confidence*100:.1f}%)')


if __name__ == '__main__':
    main()
