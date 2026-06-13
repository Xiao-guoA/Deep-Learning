"""
Streamlit Web 界面 - 人脸表情识别

用法:
    streamlit run app.py
"""

import os
import sys
import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
import streamlit as st

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models.cnn_baseline import CNNBaseline
from models.resnet_expression import ResNetExpression

# 标签映射
EMOTION_LABELS = {
    0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy',
    4: 'Sad', 5: 'Surprise', 6: 'Neutral',
}

EMOTION_EMOJIS = {
    'Angry':    '\U0001F620',
    'Disgust':  '\U0001F922',
    'Fear':     '\U0001F628',
    'Happy':    '\U0001F604',
    'Sad':      '\U0001F622',
    'Surprise': '\U0001F632',
    'Neutral':  '\U0001F610',
}

EMOTION_COLORS = {
    'Angry':    '#FF4444',
    'Disgust':  '#44AA44',
    'Fear':     '#AA44AA',
    'Happy':    '#44CC44',
    'Sad':      '#4444FF',
    'Surprise': '#FFDD44',
    'Neutral':  '#AAAAAA',
}

NORMALIZE = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
)


@st.cache_resource
def load_model_cached(model_type='cnn'):
    """缓存模型加载"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if model_type == 'cnn':
        model = CNNBaseline(num_classes=7)
        checkpoint_path = 'checkpoints/cnn/best_model.pth'
        size = 48
    else:
        model = ResNetExpression(num_classes=7, finetune_all=True)
        checkpoint_path = 'checkpoints/resnet18/best_model.pth'
        size = 224

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        NORMALIZE,
    ])

    return model, transform, device


def main():
    st.set_page_config(
        page_title='人脸表情识别',
        page_icon='\U0001F604',
        layout='wide',
    )

    st.markdown("""
    <h1 style='text-align: center; margin-bottom: 0;'>
        人脸表情识别系统
    </h1>
    <p style='text-align: center; color: gray; margin-top: 0;'>
        自建 CNN / ResNet18 迁移学习 · FER2013 数据集
    </p>
    <hr>
    """, unsafe_allow_html=True)

    # 侧边栏
    with st.sidebar:
        st.header("模型设置")
        model_type = st.radio(
            "选择模型",
            options=['cnn', 'resnet18'],
            format_func=lambda x: '自建 CNN（48x48）' if x == 'cnn' else 'ResNet18 迁移学习（224x224）',
            index=0,
        )

        st.markdown("---")
        st.header("关于")
        st.markdown(f"""
        使用 **{ '自建 CNN' if model_type == 'cnn' else 'ResNet18' }** 进行人脸表情识别。

        **可识别表情:**
        - Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral

        **技术栈:** PyTorch, OpenCV, Streamlit
        """)

        st.markdown("---")
        model_loaded = False
        try:
            model, transform, device = load_model_cached(model_type)
            model_loaded = True
            st.success(f"模型已加载 ({device})")
        except Exception as e:
            st.error(f"模型加载失败: {e}")

    tab1, tab2 = st.tabs(["图片上传", "摄像头"])

    # Tab 1: 图片上传
    with tab1:
        st.header("上传图片进行识别")
        uploaded = st.file_uploader(
            "选择图片文件",
            type=['jpg', 'jpeg', 'png', 'bmp'],
        )

        use_camera_input = st.checkbox("改为拍照", value=False)
        if use_camera_input:
            camera_photo = st.camera_input("拍照")
            if camera_photo:
                uploaded = camera_photo

        if uploaded is not None and model_loaded:
            image = Image.open(uploaded).convert('RGB')
            img_array = np.array(image)

            col1, col2 = st.columns(2)

            with col1:
                st.image(image, caption='输入图片', use_container_width=True)

            with col2:
                with st.spinner('分析中...'):
                    opencv_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                    gray = cv2.cvtColor(opencv_img, cv2.COLOR_BGR2GRAY)
                    face_cascade = cv2.CascadeClassifier(
                        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                    )
                    faces = face_cascade.detectMultiScale(
                        gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48)
                    )

                    if len(faces) > 0:
                        face = max(faces, key=lambda f: f[2] * f[3])
                        x, y, w, h = face
                        face_roi = image.crop((x, y, x + w, y + h))

                        # 推理
                        with torch.no_grad():
                            tensor = transform(face_roi).unsqueeze(0).to(device)
                            outputs = model(tensor)
                            probs = torch.softmax(outputs, dim=1)
                            confidence, predicted = torch.max(probs, dim=1)
                            confidence = confidence.item()
                            predicted = predicted.item()
                            emotion = EMOTION_LABELS[predicted]
                            probs_np = probs[0].cpu().numpy()

                        emoji = EMOTION_EMOJIS.get(emotion, '')
                        color = EMOTION_COLORS.get(emotion, '#000000')

                        st.markdown(
                            f"<div style='text-align: center; padding: 20px; "
                            f"border-radius: 10px; background-color: {color}22; "
                            f"border: 2px solid {color};'>"
                            f"<h2 style='font-size: 64px; margin: 0;'>{emoji}</h2>"
                            f"<h2>识别结果: {emotion}</h2>"
                            f"<h3 style='color: gray;'>置信度: "
                            f"{confidence*100:.1f}%</h3></div>",
                            unsafe_allow_html=True,
                        )

                        # 各类概率
                        st.subheader("各类别概率")
                        probs_dict = {EMOTION_LABELS[i]: float(probs_np[i]) for i in range(7)}
                        sorted_probs = sorted(probs_dict.items(), key=lambda x: x[1], reverse=True)
                        for emotion_name, prob in sorted_probs:
                            st.markdown(
                                f"{EMOTION_EMOJIS.get(emotion_name, '')} "
                                f"**{emotion_name}**: {prob*100:.1f}%"
                            )
                            st.progress(prob)
                    else:
                        st.error("未检测到人脸，请换一张图片重试。")

    # Tab 2: 摄像头
    with tab2:
        st.header("实时摄像头识别")
        st.markdown("""
        如需更好的实时性能，建议使用专用脚本：

        ```bash
        python realtime_demo.py --model {model_type}
        ```
        """)

        run_camera = st.checkbox("开启快照模式", value=False)
        if run_camera and model_loaded:
            FRAME_WINDOW = st.image([], caption='摄像头', width=640)
            cap = cv2.VideoCapture(0)
            stop = st.button("停止")

            if not cap.isOpened():
                st.error("无法打开摄像头")
            else:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    face_cascade = cv2.CascadeClassifier(
                        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                    )
                    faces = face_cascade.detectMultiScale(
                        gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48)
                    )

                    for (x, y, w, h) in faces:
                        face_roi = frame[y:y+h, x:x+w]
                        face_pil = Image.fromarray(cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB))
                        try:
                            tensor = transform(face_pil).unsqueeze(0).to(device)
                            with torch.no_grad():
                                outputs = model(tensor)
                                probs = torch.softmax(outputs, dim=1)
                                confidence, predicted = torch.max(probs, dim=1)
                                emotion = EMOTION_LABELS[predicted.item()]

                            color = EMOTION_COLORS.get(emotion, (0, 255, 0))
                            if isinstance(color, str):
                                color = tuple(int(color[i:i+2], 16) for i in range(1, 6, 2))
                                color = (color[2], color[1], color[0])
                            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                            cv2.putText(frame, f"{emotion} ({confidence*100:.1f}%)",
                                       (x, y-10), cv2.FONT_HERSHEY_SIMPLEX,
                                       0.6, (0, 255, 0), 2)
                        except Exception:
                            pass

                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    FRAME_WINDOW.image(frame_rgb)

                    if stop:
                        break

                cap.release()


if __name__ == '__main__':
    main()
