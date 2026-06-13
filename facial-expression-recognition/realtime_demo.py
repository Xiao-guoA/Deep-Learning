"""
摄像头实时人脸表情识别

使用 OpenCV 进行人脸检测，自建 CNN 或 ResNet18 进行表情分类。
按 'q' 退出，按 's' 保存截图。

用法:
    # 自建 CNN（默认）
    python realtime_demo.py

    # ResNet18
    python realtime_demo.py --model resnet18
"""

import sys
import os
import argparse
import cv2
import torch
import numpy as np
from collections import deque
from PIL import Image
from torchvision import transforms

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 标签映射
EMOTION_LABELS = {
    0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy',
    4: 'Sad', 5: 'Surprise', 6: 'Neutral',
}

# 标准归一化
NORMALIZE = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
)

# 各类别显示颜色
EMOTION_COLORS = {
    'Angry':    (0, 0, 255),
    'Disgust':  (0, 128, 0),
    'Fear':     (128, 0, 128),
    'Happy':    (0, 255, 0),
    'Sad':      (255, 0, 0),
    'Surprise': (0, 255, 255),
    'Neutral':  (255, 255, 255),
}


def get_transform(model_type):
    """根据模型类型获取预处理 transform"""
    size = 48 if model_type == 'cnn' else 224
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        NORMALIZE,
    ])


def load_model(model_type='cnn', checkpoint_path=None, device='cuda'):
    """加载模型"""
    if model_type == 'cnn':
        from models.cnn_baseline import CNNBaseline
        model = CNNBaseline(num_classes=7)
    else:
        from models.resnet_expression import ResNetExpression
        model = ResNetExpression(num_classes=7, finetune_all=True)

    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()
    return model


@torch.no_grad()
def predict(model, face_pil, transform, device):
    """预测单张人脸的表情"""
    tensor = transform(face_pil).unsqueeze(0).to(device)
    outputs = model(tensor)
    probs = torch.softmax(outputs, dim=1)
    confidence, predicted = torch.max(probs, dim=1)
    probs_np = probs[0].cpu().numpy()
    probs_dict = {EMOTION_LABELS[i]: float(probs_np[i]) for i in range(7)}
    return EMOTION_LABELS[predicted.item()], confidence.item(), probs_dict


def draw_face_info(frame, x, y, w, h, emotion, confidence):
    """绘制人脸框和表情标签"""
    color = EMOTION_COLORS.get(emotion, (255, 255, 255))
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    label = f"{emotion} ({confidence*100:.1f}%)"
    size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
    cv2.rectangle(frame, (x, y - 30), (x + size[0] + 10, y), color, -1)
    cv2.putText(frame, label, (x + 5, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)


def main(args):
    device = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
    print(f'设备: {device}')
    print(f'模型: {args.model}')

    # 默认 checkpoint 路径
    if args.checkpoint is None:
        args.checkpoint = f'checkpoints/{args.model}/best_model.pth'

    model = load_model(args.model, args.checkpoint, device)
    transform = get_transform(args.model)

    # 加载人脸检测器
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        print('错误: 无法加载人脸检测模型')
        return

    pred_buffer = deque(maxlen=args.smooth_frames)
    current_emotion = '检测中...'
    current_confidence = 0.0

    cap = cv2.VideoCapture(args.camera_id)
    if not cap.isOpened():
        print(f'错误: 无法打开摄像头 (ID: {args.camera_id})')
        return

    print(f'\n摄像头已打开。按 q 退出，按 s 保存截图。\n')

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        display = frame.copy()

        # 半分辨率人脸检测（提速）
        h, w = frame.shape[:2]
        small = cv2.resize(frame, (w // 2, h // 2))
        gray_small = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray_small, scaleFactor=1.1, minNeighbors=5, minSize=(24, 24)
        )

        if len(faces) > 0:
            # 选最大的人脸
            face = max(faces, key=lambda f: f[2] * f[3])
            x, y, fw, fh = face
            x, y, fw, fh = x * 2, y * 2, fw * 2, fh * 2

            if frame_count % args.infer_interval == 0:
                try:
                    face_roi = frame[y:y+fh, x:x+fw]
                    face_pil = Image.fromarray(cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB))
                    emotion, confidence, probs = predict(model, face_pil, transform, device)
                    pred_buffer.append((emotion, confidence, probs))

                    # 移动平均平滑
                    if pred_buffer:
                        avg_probs = {}
                        for p in pred_buffer:
                            for emo, prob in p[2].items():
                                avg_probs.setdefault(emo, []).append(prob)
                        avg = {emo: np.mean(vals) for emo, vals in avg_probs.items()}
                        current_emotion = max(avg, key=avg.get)
                        current_confidence = avg[current_emotion]

                except Exception as e:
                    print(f'推理错误: {e}')

            draw_face_info(display, x, y, fw, fh, current_emotion, current_confidence)

        cv2.putText(display, f'Faces: {len(faces)}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow('Real-time Facial Expression Recognition', display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            cv2.imwrite(f'screenshot_{frame_count}.jpg', display)
            print(f'Screenshot saved: screenshot_{frame_count}.jpg')

    cap.release()
    cv2.destroyAllWindows()
    print('摄像头已关闭。')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='摄像头实时表情识别')
    parser.add_argument('--model', type=str, default='cnn',
                        choices=['cnn', 'resnet18'],
                        help='模型类型: cnn / resnet18')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='模型权重路径')
    parser.add_argument('--camera-id', type=int, default=0,
                        help='摄像头设备 ID')
    parser.add_argument('--cpu', action='store_true',
                        help='强制使用 CPU')
    parser.add_argument('--smooth-frames', type=int, default=5,
                        help='移动平均窗口大小')
    parser.add_argument('--infer-interval', type=int, default=3,
                        help='每隔 N 帧推理一次')

    args = parser.parse_args()
    main(args)
