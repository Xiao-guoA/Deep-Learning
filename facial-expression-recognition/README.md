# 人脸表情识别系统

基于 Vision Transformer（ViT）的实时人脸表情识别。模型来自 Hugging Face Hub，在 FER2013 数据集上微调，下载后即可使用，无需训练。

## 功能

- 图片上传与表情分类（支持单张和批量）
- 实时摄像头推理
- 支持 7 类表情：生气、厌恶、恐惧、开心、悲伤、惊讶、中性
- Streamlit Web 界面
- GPU 加速（自动回退到 CPU）

## 项目结构

```
facial-expression-recognition/
├── models/
│   └── vit_expression.py       # ViT 模型封装
├── utils/
│   └── dataset.py              # 标签定义
├── inference.py                # 单张/批量图片推理
├── realtime_demo.py            # 摄像头实时识别
├── app.py                      # Streamlit Web 界面
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
| 准确率 | 71.55% |
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
