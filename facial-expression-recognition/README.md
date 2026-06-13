# 人脸表情识别系统

基于自建 CNN 和 ResNet18 迁移学习的人脸表情识别系统。在 FER2013 数据集上训练，实现了 7 类基本表情的分类。

## 项目亮点

- **自建 CNN 基线**：从零设计训练（5 层卷积，~55 万参数），不依赖任何预训练权重
- **ResNet18 迁移学习**：使用 ImageNet 预训练权重进行全模型微调
- **双模型对比**：从准确率、收敛速度、过拟合等维度对比分析
- **实用部署**：支持图片、摄像头实时、Web 界面三种使用方式

## 实验效果

| 模型 | 参数量 | 测试准确率 | 训练策略 |
|------|--------|-----------|---------|
| 自建 CNN | ~55 万 | ~63% | 从零训练 20 epoch |
| ResNet18 | ~1117 万 | ~69% | 迁移学习 15 epoch |

## 项目结构

```
facial-expression-recognition/
├── checkpoints/                # 训练结果（权重、曲线、混淆矩阵）
│   ├── cnn/                    # CNN 基线
│   └── resnet18/               # ResNet18 迁移学习
├── models/
│   ├── cnn_baseline.py         # 自建 CNN 模型
│   └── resnet_expression.py    # ResNet18 迁移学习模型
├── utils/
│   └── dataset.py              # FER2013 数据加载和预处理
├── train.py                    # 训练脚本
├── inference.py                # 单张/批量图片推理
├── realtime_demo.py            # 摄像头实时识别
├── app.py                      # Streamlit Web 界面
├── 20234080327-赵果剑-课程设计.ipynb  # 课程设计报告
├── requirements.txt
└── README.md
```

## 环境要求

- Python 3.9+
- PyTorch 2.0+
- 约 500MB 磁盘空间

## 安装

```bash
cd facial-expression-recognition
pip install -r requirements.txt
```

## 训练

先下载 FER2013 数据集：[Kaggle FER2013](https://www.kaggle.com/datasets/nicolejyt/facialexpressionrecognition)

### 训练自建 CNN

```bash
python train.py --csv 路径/fer2013.csv --model cnn --epochs 20
```

### 训练 ResNet18 迁移学习

```bash
# 全模型微调（推荐）
python train.py --csv 路径/fer2013.csv --model resnet18 --epochs 15

# 只训练分类头（冻结主干）
python train.py --csv 路径/fer2013.csv --model resnet18 --epochs 10 --freeze-backbone
```

训练过程中自动保存最佳模型、Loss/Accuracy 曲线、混淆矩阵和分类报告。

## 使用方法

### 图片推理

```bash
# CNN（默认）
python inference.py --image 图片路径.jpg

# ResNet18
python inference.py --image 图片路径.jpg --model resnet18

# 批量处理
python inference.py --image-dir 图片目录/ --model resnet18
```

### 摄像头实时

```bash
python realtime_demo.py
```

快捷键：`q` 退出，`s` 保存截图。

### Web 界面

```bash
streamlit run app.py
```

启动后浏览器打开 http://localhost:8501。支持图片上传和摄像头两种模式。

## 模型结构说明

### 自建 CNN

```
输入 (48x48 RGB)
  ├─ Conv(3→64) → BN → ReLU → Conv(64→64) → BN → ReLU → MaxPool → 24x24
  ├─ Conv(64→128) → BN → ReLU → Conv(128→128) → BN → ReLU → MaxPool → 12x12
  ├─ Conv(128→256) → BN → ReLU → MaxPool → 6x6
  └─ AdaptiveAvgPool → Dropout → FC(256→7)
```

全部使用 3×3 卷积 + Kaiming 初始化。

### ResNet18

使用 torchvision 预训练权重，替换分类头为 `FC(512→256) → ReLU → FC(256→7)`。支持全模型微调和只训练分类头两种策略。

## 数据集说明

FER2013：35,887 张 48×48 灰度图，7 类表情。其中 Disgust 仅 547 张，存在类别不平衡。

| 表情 | 训练集 | 测试集 |
|------|--------|--------|
| Angry | 3,995 | 491 |
| Disgust | 436 | 56 |
| Fear | 4,097 | 528 |
| Happy | 7,215 | 879 |
| Sad | 4,845 | 579 |
| Surprise | 3,171 | 416 |
| Neutral | 4,950 | 641 |

## 注意事项

- 首次运行 ResNet18 会从 torchvision 下载预训练权重（约 45MB）
- 训练数据和结果默认保存在 `checkpoints/` 目录，已加入 `.gitignore`
- 人脸检测使用 OpenCV Haar Cascade
- 已在 Python 3.12 + PyTorch 2.5 + CUDA 12.1 环境下测试

## 参考

- [FER2013 数据集](https://www.kaggle.com/datasets/nicolejyt/facialexpressionrecognition)
- [Deep Residual Learning (ResNet)](https://arxiv.org/abs/1512.03385) (He et al., 2015)
- [Very Deep Convolutional Networks (VGG)](https://arxiv.org/abs/1409.1556) (Simonyan & Zisserman, 2014)
