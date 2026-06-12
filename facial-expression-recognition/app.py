"""
Streamlit web application for facial expression recognition.

Run with: streamlit run app.py
"""

import os
import sys
import cv2
import numpy as np
import torch
from PIL import Image
import streamlit as st

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from inference import load_model, vit_preprocess, predict

# Emotion to emoji mapping for display
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


@st.cache_resource
def load_model_cached():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return load_model(device)


def main():
    st.set_page_config(
        page_title='Facial Expression Recognition',
        page_icon='\U0001F604',
        layout='wide',
    )

    st.markdown("""
    <h1 style='text-align: center; margin-bottom: 0;'>
        Facial Expression Recognition
    </h1>
    <p style='text-align: center; color: gray; margin-top: 0;'>
        Vision Transformer fine-tuned on FER2013
    </p>
    <hr>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.header("About")
        st.markdown("""
        This application uses a Vision Transformer (ViT) model
        pre-trained on ImageNet and fine-tuned on the FER2013 dataset.

        **Recognized expressions:**
        - Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral

        **Model:** abhilash88/face-emotion-detection (71.55% accuracy)

        **Tech stack:** PyTorch, Hugging Face Transformers,
        OpenCV, Streamlit
        """)

        st.markdown("---")

        model_loaded = False
        try:
            model = load_model_cached()
            model_loaded = True
            st.success("Model loaded")
        except Exception as e:
            st.error(f"Failed to load model: {e}")

    tab1, tab2 = st.tabs(["Image Upload", "Camera"])

    # Tab 1: Image upload
    with tab1:
        st.header("Upload an image")
        uploaded = st.file_uploader(
            "Choose an image file",
            type=['jpg', 'jpeg', 'png', 'bmp'],
        )

        use_camera_input = st.checkbox("Take a photo instead", value=False)
        if use_camera_input:
            camera_photo = st.camera_input("Take a photo")
            if camera_photo:
                uploaded = camera_photo

        if uploaded is not None and model_loaded:
            image = Image.open(uploaded).convert('RGB')
            img_array = np.array(image)

            col1, col2 = st.columns(2)

            with col1:
                st.image(image, caption='Input image', use_container_width=True)

            with col2:
                with st.spinner('Analyzing...'):
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
                        tensor = vit_preprocess(face_roi)

                        device = 'cuda' if torch.cuda.is_available() else 'cpu'
                        emotion, confidence, probs = predict(model, tensor, device)

                        emoji = EMOTION_EMOJIS.get(emotion, '')
                        color = EMOTION_COLORS.get(emotion, '#000000')

                        st.markdown(
                            f"<div style='text-align: center; padding: 20px; "
                            f"border-radius: 10px; background-color: {color}22; "
                            f"border: 2px solid {color};'>"
                            f"<h2 style='font-size: 64px; margin: 0;'>{emoji}</h2>"
                            f"<h2>Result: {emotion}</h2>"
                            f"<h3 style='color: gray;'>Confidence: "
                            f"{confidence*100:.1f}%</h3></div>",
                            unsafe_allow_html=True,
                        )

                        # Draw bounding box on the image
                        result = np.array(image).copy()
                        cv2.rectangle(result, (x, y), (x + w, y + h),
                                     tuple(int(color[i:i+2], 16) for i in range(1, 6, 2)), 2)

                        st.subheader("Per-class probabilities")
                        sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
                        for emotion_name, prob in sorted_probs:
                            st.markdown(
                                f"{EMOTION_EMOJIS.get(emotion_name, '')} "
                                f"**{emotion_name}**: {prob*100:.1f}%"
                            )
                            st.progress(prob)
                    else:
                        st.error("No face detected. Try a different image.")

    # Tab 2: Camera (simplified - recommends realtime_demo.py)
    with tab2:
        st.header("Real-time Camera")
        st.markdown("""
        For real-time camera inference with better performance,
        use the dedicated script:

        ```
        python realtime_demo.py
        ```

        This Streamlit tab provides a simplified snapshot-based
        alternative.
        """)

        run_camera = st.checkbox("Enable snapshot mode", value=False)
        if run_camera and model_loaded:
            FRAME_WINDOW = st.image([], caption='Camera', width=640)
            cap = cv2.VideoCapture(0)
            stop = st.button("Stop")

            if not cap.isOpened():
                st.error("Cannot open camera")
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
                            tensor = vit_preprocess(face_pil)
                            device = 'cuda' if torch.cuda.is_available() else 'cpu'
                            emotion, confidence, _ = predict(model, tensor, device)
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
