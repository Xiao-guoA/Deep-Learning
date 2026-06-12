# 人脸表情识别系统

基于 Vision Transformer（ViT）的实时人脸表情识别。在 FER2013 数据集上对 ViT-Base 进行微调，测试集准确率约 72%。

## 功能

- 模型微调（迁移学习，10 epoch 约 20-30 分钟）
- 图片上传与表情分类（支持单张和批量）
- 实时摄像头推理
- 注意力可视化
- Streamlit Web 界面
- GPU 加速（自动回退到 CPU）

## 项目结构

```
facial-expression-recognition/
├── checkpoints/                # 训练保存的模型和曲线
├── models/
│   └── vit_expression.py       # ViT 模型封装
├── utils/
│   └── dataset.py              # FER2013 数据集加载和预处理
├── train.py                    # 微调训练脚本
├── inference.py                # 单张/批量图片推理
├── realtime_demo.py            # 摄像头实时识别
├── app.py                      # Streamlit Web 界面
├── DL课程设计.ipynb            # 课程设计报告（含完整代码）
├── requirements.txt
└── README.md
```

## 环境要求

- Python 3.9+
- PyTorch 2.0+
- 约 500MB 磁盘空间（模型缓存）

## 安装

```bash
cd facial-expression-recognition
pip install -r requirements.txt
```

首次运行时会自动从 Hugging Face Hub 下载 ViT 模型（约 330MB）。

## 模型微调

在 FER2013 数据集上对 ViT 进行迁移学习（需要先下载 fer2013.csv）：

```bash
# Kaggle 下载: https://www.kaggle.com/datasets/nicolejyt/facialexpressionrecognition
# 或手动下载后放到本地路径

python train.py --csv 你的路径/fer2013.csv --epochs 10
```

参数说明：
- `--epochs` 训练轮数，默认 10
- `--lr` 学习率，默认 5e-5
- `--batch-size` 批次大小，默认 64
- `--output-dir` 输出目录，默认 checkpoints/

训练完成后在 `checkpoints/` 下生成：
- `best_model.pth` — 最佳模型权重
- `training_curves.png` — Loss 和 Accuracy 曲线
- `confusion_matrix.png` — 混淆矩阵
- `classification_report.txt` — 分类报告

## 使用方法

### 图片推理

单张图片：

```bash
python inference.py --image 图片路径.jpg
```

批量处理目录下的所有图片：

```bash
python inference.py --image-dir 图片目录/
```

### 实时摄像头

```bash
python realtime_demo.py
```

快捷键：
- `q` 退出
- `s` 保存截图

指定摄像头或强制使用 CPU：

```bash
python realtime_demo.py --camera-id 1 --cpu
```

### Web 界面

```bash
streamlit run app.py
```

启动后浏览器打开 http://localhost:8501 即可使用。

## 模型说明

| 项目 | 说明 |
|------|------|
| 模型结构 | Vision Transformer (ViT-Base) |
| 模型来源 | abhilash88/face-emotion-detection |
| 训练数据 | FER2013 |
| 微调方式 | 全模型微调 (Full fine-tuning) |
| 优化器 | AdamW (lr=5e-5, weight_decay=0.01) |
| 学习率调度 | Cosine decay with 10% warmup |
| 数据增强 | RandomHorizontalFlip, RandomRotation |
| 准确率 | ~72% (FER2013 PrivateTest) |
| 参数量 | 8580 万 |
| 输入尺寸 | 224x224 |

模型通过 Hugging Face `transformers` 库加载，首次下载后会在本地缓存。

## 注意事项

- 首次运行需要下载模型（约 330MB），仅需一次
- 有 CUDA 显卡时自动使用 GPU 加速
- 人脸检测使用 OpenCV Haar Cascade
- 已在 Python 3.12 + PyTorch 2.5 + CUDA 12.1 环境下测试

## 参考

- [FER2013 数据集](https://www.kaggle.com/datasets/nicolejyt/facialexpressionrecognition)
- [Hugging Face 模型页](https://huggingface.co/abhilash88/face-emotion-detection)
- [ViT 论文](https://arxiv.org/abs/2010.11929) (Dosovitskiy et al., 2021)
