# Deep-Learning

深度学习课程作业与项目。

## 内容

| 目录/文件 | 说明 |
|-----------|------|
| `HW01-*.ipynb` ~ `HW04-*.ipynb` | 课程作业（Jupyter Notebook） |
| `facial-expression-recognition/` | 课程项目：基于自建 CNN 和 ResNet18 迁移学习的实时人脸表情识别系统 |

## 项目：人脸表情识别系统

在 FER2013 数据集上训练自建 CNN（从零训练）和 ResNet18（迁移学习），实现 7 类表情识别。

| 模型 | 参数量 | 测试准确率 |
|------|--------|-----------|
| 自建 CNN（5 层卷积） | ~55 万 | ~63% |
| ResNet18 迁移学习 | ~1117 万 | ~69% |

### 快速开始

```bash
cd facial-expression-recognition
pip install -r requirements.txt
```

图片推理：

```bash
# 自建 CNN（默认）
python inference.py --image 图片路径.jpg
# ResNet18
python inference.py --image 图片路径.jpg --model resnet18
```

摄像头实时识别：

```bash
python realtime_demo.py
```

Web 界面：

```bash
streamlit run app.py
```

详细说明见项目内的 [README.md](facial-expression-recognition/README.md)。

### 技术要点

- 模型：自建 CNN + ResNet18 迁移学习（torchvision 预训练权重）
- 数据集：FER2013（35,887 张 48×48 灰度图，7 类表情）
- 人脸检测：OpenCV Haar Cascade
- Web 界面：Streamlit

## 环境

- Python 3.9+
- PyTorch 2.0+
- 各项目依赖见对应目录下的 requirements.txt
