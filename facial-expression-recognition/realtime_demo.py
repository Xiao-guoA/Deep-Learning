"""
Real-time facial expression recognition via webcam.

Uses OpenCV for face detection and a pre-trained Vision Transformer
for expression classification. Press 'q' to quit, 's' to save a screenshot.
"""

import sys
import os
import argparse
import cv2
import torch
import numpy as np
from collections import deque
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference import load_model, vit_preprocess, predict

# Display colors per emotion
EMOTION_COLORS = {
    'Angry':    (0, 0, 255),
    'Disgust':  (0, 128, 0),
    'Fear':     (128, 0, 128),
    'Happy':    (0, 255, 0),
    'Sad':      (255, 0, 0),
    'Surprise': (0, 255, 255),
    'Neutral':  (255, 255, 255),
}


def draw_face_info(frame, x, y, w, h, emotion, confidence):
    """Draw bounding box and emotion label on the frame."""
    color = EMOTION_COLORS.get(emotion, (255, 255, 255))
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    label = f"{emotion} ({confidence*100:.1f}%)"
    size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
    cv2.rectangle(frame, (x, y - 30), (x + size[0] + 10, y), color, -1)
    cv2.putText(frame, label, (x + 5, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)


def main(args):
    device = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
    print(f"Device: {device}")

    model = load_model(device)

    # Initialize face detector
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        print("Error: failed to load face detection model")
        return

    pred_buffer = deque(maxlen=args.smooth_frames)
    current_emotion = "Detecting..."
    current_confidence = 0.0

    cap = cv2.VideoCapture(args.camera_id)
    if not cap.isOpened():
        print(f"Error: cannot open camera (ID: {args.camera_id})")
        return

    print(f"\nCamera opened. Press 'q' to quit, 's' to save screenshot.\n")

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        display = frame.copy()

        # Face detection at half resolution for speed
        h, w = frame.shape[:2]
        small = cv2.resize(frame, (w // 2, h // 2))
        gray_small = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray_small, scaleFactor=1.1, minNeighbors=5, minSize=(24, 24)
        )

        if len(faces) > 0:
            # Scale coordinates back to original resolution
            face = max(faces, key=lambda f: f[2] * f[3])
            x, y, fw, fh = face
            x, y, fw, fh = x * 2, y * 2, fw * 2, fh * 2

            if frame_count % args.infer_interval == 0:
                try:
                    face_roi = frame[y:y+fh, x:x+fw]
                    face_pil = Image.fromarray(cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB))
                    tensor = vit_preprocess(face_pil, device=device)
                    emotion, confidence, probs = predict(model, tensor, device)
                    pred_buffer.append((emotion, confidence, probs))

                    if pred_buffer:
                        avg_probs = {}
                        for p in pred_buffer:
                            for emo, prob in p[2].items():
                                avg_probs.setdefault(emo, []).append(prob)
                        avg = {emo: np.mean(vals) for emo, vals in avg_probs.items()}
                        current_emotion = max(avg, key=avg.get)
                        current_confidence = avg[current_emotion]

                except Exception as e:
                    print(f"Inference error: {e}")

            draw_face_info(display, x, y, fw, fh, current_emotion, current_confidence)

        cv2.putText(display, f"Faces: {len(faces)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow('Real-time Facial Expression Recognition', display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            cv2.imwrite(f"screenshot_{frame_count}.jpg", display)
            print(f"Screenshot saved: screenshot_{frame_count}.jpg")

    cap.release()
    cv2.destroyAllWindows()
    print("Camera closed.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Real-time facial expression recognition')
    parser.add_argument('--camera-id', type=int, default=0, help='Camera device ID')
    parser.add_argument('--cpu', action='store_true', help='Force CPU inference')
    parser.add_argument('--smooth-frames', type=int, default=5,
                        help='Moving average window size')
    parser.add_argument('--infer-interval', type=int, default=3,
                        help='Run inference every N frames')

    args = parser.parse_args()
    main(args)
