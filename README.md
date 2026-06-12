# Deep-Learning

深度学习课程作业与项目。

## 内容

| 目录/文件 | 说明 |
|-----------|------|
| `HW01-*.ipynb` ~ `HW03-*.ipynb` | 课程作业（Jupyter Notebook） |
| `facial-expression-recognition/` | 课程项目：基于 ViT 的实时人脸表情识别系统 |

## 项目：人脸表情识别系统

基于 Vision Transformer（ViT）的实时人脸表情识别，支持 7 类表情（生气、厌恶、恐惧、开心、悲伤、惊讶、中性）。

### 快速开始

```bash
cd facial-expression-recognition
pip install -r requirements.txt
```

图片推理：

```bash
python inference.py --image 图片路径.jpg
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

- 模型：Hugging Face 预训练 ViT（abhilash88/face-emotion-detection）
- 数据集：FER2013（71.55% 准确率）
- 人脸检测：OpenCV Haar Cascade
- Web 界面：Streamlit

## 环境

- Python 3.9+
- PyTorch 2.0+
- 各项目依赖见对应目录下的 requirements.txt
