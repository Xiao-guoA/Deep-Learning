"""
FER2013 模型训练脚本

支持两种模型架构：
  1. cnn      — 自建 CNN 基线模型（从零训练，~55 万参数）
  2. resnet18 — ResNet18 迁移学习（基于 ImageNet 预训练权重）

用法:
    # 训练自建 CNN
    python train.py --csv E:/下载/archive/fer2013.csv --model cnn --epochs 20

    # 训练 ResNet18（全模型微调）
    python train.py --csv E:/下载/archive/fer2013.csv --model resnet18 --epochs 15

    # 训练 ResNet18（只训练分类头，冻结主干）
    python train.py --csv E:/下载/archive/fer2013.csv --model resnet18 --freeze-backbone
"""

import os
import sys
import argparse
import json
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.dataset import get_fer2013_dataloaders, EMOTION_LABELS, NUM_CLASSES


def train_one_epoch(model, loader, criterion, optimizer, device):
    """训练一个 epoch，返回平均 loss 和准确率"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        if (batch_idx + 1) % 100 == 0:
            print(f'  Batch {batch_idx+1}/{len(loader)} | Loss: {loss.item():.4f}')

    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """评估模型，返回 loss、准确率、预测结果和真实标签"""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    avg_loss = running_loss / len(all_labels)
    accuracy = accuracy_score(all_labels, all_preds)

    return avg_loss, accuracy, all_preds, all_labels


def plot_training_curves(train_losses, val_losses, train_accs, val_accs, save_path):
    """绘制训练曲线"""
    epochs = range(1, len(train_losses) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, train_losses, 'b-', label='Train Loss')
    ax1.plot(epochs, val_losses, 'r-', label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss 曲线')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, train_accs, 'b-', label='Train Acc')
    ax2.plot(epochs, val_accs, 'r-', label='Val Acc')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Accuracy 曲线')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f'训练曲线已保存: {save_path}')
    plt.close()


def plot_confusion_matrix(y_true, y_pred, labels, save_path):
    """绘制混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))

    import seaborn as sns
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix (FER2013 Test)')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f'混淆矩阵已保存: {save_path}')
    plt.close()

    return cm


def create_model(model_type, num_classes, freeze_backbone, device):
    """根据参数创建模型实例"""
    if model_type == 'cnn':
        from models.cnn_baseline import CNNBaseline, count_parameters
        model = CNNBaseline(num_classes=num_classes)
        print(f'模型: 自建 CNN 基线 ({count_parameters(model):,} 参数)')
        print(f'  输入尺寸: 48x48 (原始 FER2013 分辨率)')

    elif model_type == 'resnet18':
        from models.resnet_expression import ResNetExpression, count_parameters
        model = ResNetExpression(num_classes=num_classes,
                                  finetune_all=not freeze_backbone)
        if freeze_backbone:
            print('策略: 冻结主干 → 只训练分类头')
        else:
            print('策略: 全模型微调')
        total = sum(p.numel() for p in model.parameters())
        trainable = count_parameters(model)
        print(f'  总参数量: {total:,}, 可训练参数: {trainable:,}')
        print(f'  输入尺寸: 224x224 (ResNet 标准输入)')

    else:
        raise ValueError(f'未知模型类型: {model_type}')

    return model.to(device)


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'设备: {device}')
    print(f'CSV 路径: {args.csv}')
    print(f'模型: {args.model}, Epochs: {args.epochs}, Batch: {args.batch_size}, LR: {args.lr}')
    print()

    # 创建输出目录
    output_dir = os.path.join(args.output_dir, args.model)
    os.makedirs(output_dir, exist_ok=True)

    # 1. 数据加载
    print('=' * 50)
    print('加载数据集...')
    model_type_for_data = 'cnn' if args.model == 'cnn' else 'resnet'
    train_loader, val_loader, test_loader = get_fer2013_dataloaders(
        csv_path=args.csv,
        batch_size=args.batch_size,
        model_type=model_type_for_data,
        num_workers=args.num_workers,
    )
    print(f'训练集: {len(train_loader.dataset)} 张')
    print(f'验证集: {len(val_loader.dataset)} 张')
    print(f'测试集: {len(test_loader.dataset)} 张')
    print()

    # 2. 创建模型
    print('=' * 50)
    print('创建模型...')
    model = create_model(args.model, NUM_CLASSES, args.freeze_backbone, device)
    print()

    # 3. 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # 余弦退火学习率调度
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    print(f'优化器: AdamW (lr={args.lr}, weight_decay={args.weight_decay})')
    print(f'调度器: CosineAnnealingLR (T_max={args.epochs})')
    print()

    # 4. 训练循环
    print('=' * 50)
    print('开始训练...')
    best_val_acc = 0.0
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(1, args.epochs + 1):
        print(f'\nEpoch {epoch}/{args.epochs}')
        epoch_start = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)

        scheduler.step()
        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]['lr']

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}%')
        print(f'  Val   Loss: {val_loss:.4f}, Val   Acc: {val_acc*100:.2f}%')
        print(f'  Time: {epoch_time:.1f}s, LR: {current_lr:.2e}')

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
            }
            torch.save(checkpoint, os.path.join(output_dir, 'best_model.pth'))
            print(f'  -> 保存最佳模型 (Val Acc: {val_acc*100:.2f}%)')

    print(f'\n训练完成！最佳验证准确率: {best_val_acc*100:.2f}%')

    # 5. 保存训练曲线
    plot_training_curves(
        train_losses, val_losses, train_accs, val_accs,
        os.path.join(output_dir, 'training_curves.png')
    )

    # 6. 加载最佳模型，测试集评估
    print('\n' + '=' * 50)
    print('测试集评估...')
    checkpoint = torch.load(os.path.join(output_dir, 'best_model.pth'),
                            map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    test_loss, test_acc, y_pred, y_true = evaluate(model, test_loader, criterion, device)
    print(f'测试集 Loss: {test_loss:.4f}')
    print(f'测试集准确率: {test_acc*100:.2f}%')

    # 分类报告
    labels_list = [EMOTION_LABELS[i] for i in range(NUM_CLASSES)]
    report = classification_report(y_true, y_pred, target_names=labels_list, digits=4)
    print('\n分类报告:')
    print(report)

    # 保存分类报告
    report_path = os.path.join(output_dir, 'classification_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f'Test Accuracy: {test_acc*100:.2f}%\n\n')
        f.write(report)
    print(f'分类报告已保存: {report_path}')

    # 混淆矩阵
    cm_path = os.path.join(output_dir, 'confusion_matrix.png')
    plot_confusion_matrix(y_true, y_pred, labels_list, cm_path)

    # 保存训练历史
    history = {
        'model': args.model,
        'train_losses': [float(x) for x in train_losses],
        'val_losses': [float(x) for x in val_losses],
        'train_accs': [float(x) for x in train_accs],
        'val_accs': [float(x) for x in val_accs],
        'test_accuracy': float(test_acc),
        'best_val_accuracy': float(best_val_acc),
        'epochs': args.epochs,
    }
    with open(os.path.join(output_dir, 'history.json'), 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2)
    print(f'训练历史已保存')

    print(f'\n所有结果已保存到: {os.path.abspath(output_dir)}')
    print('Done.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FER2013 表情识别训练')
    parser.add_argument('--csv', type=str, required=True,
                        help='fer2013.csv 文件路径')
    parser.add_argument('--model', type=str, default='cnn',
                        choices=['cnn', 'resnet18'],
                        help='模型架构: cnn (自建) / resnet18 (迁移学习)')
    parser.add_argument('--epochs', type=int, default=20,
                        help='训练轮数 (默认: 20)')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='批次大小 (默认: 64)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='学习率 (默认: 1e-3)')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='权重衰减 (默认: 1e-4)')
    parser.add_argument('--freeze-backbone', action='store_true',
                        help='冻结 ResNet 主干，只训练分类头（仅 resnet18 有效）')
    parser.add_argument('--num-workers', type=int, default=0,
                        help='数据加载线程数，Windows 建议 0 (默认: 0)')
    parser.add_argument('--output-dir', type=str, default='checkpoints',
                        help='输出目录 (默认: checkpoints)')

    args = parser.parse_args()
    main(args)
