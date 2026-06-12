"""
Facial expression recognition inference using a pre-trained Vision Transformer.

Supports single image and batch image inference.
The model is loaded from Hugging Face Hub (abhilash88/face-emotion-detection),
fine-tuned on FER2013 with 71.55% accuracy.
"""

import os
import sys
import argparse
import torch
import numpy as np
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.vit_expression import ViTExpression, vit_preprocess


def load_model(device='cuda'):
    """Load the pre-trained ViT model from Hugging Face Hub."""
    print("Loading ViT model from Hugging Face Hub...")
    model = ViTExpression()
    model = model.to(device)
    model.eval()
    print(f"Model loaded. Parameters: {sum(p.numel() for p in model.parameters()):,}")
    return model


@torch.no_grad()
def predict(model, image_tensor, device='cuda'):
    """
    Predict facial expression for a single image.

    Args:
        model: ViTExpression model.
        image_tensor: torch.Tensor, shape (1, 3, 224, 224).
        device: 'cuda' or 'cpu'.

    Returns:
        emotion: str, predicted emotion label.
        confidence: float, softmax confidence.
        probs: dict, per-class probabilities.
    """
    image_tensor = image_tensor.to(device) if image_tensor.device.type != device else image_tensor
    outputs = model(image_tensor)
    probabilities = torch.softmax(outputs, dim=1)

    confidence, predicted = torch.max(probabilities, dim=1)
    confidence = confidence.item()
    predicted = predicted.item()

    emotion = model.get_label_name(predicted)
    probs = probabilities[0].cpu().numpy()
    probs_dict = {model.get_label_name(i): float(probs[i]) for i in range(len(probs))}

    return emotion, confidence, probs_dict


def predict_image(model, image_path, device='cuda'):
    """Predict expression for a single image file."""
    image = Image.open(image_path).convert('RGB')
    tensor = vit_preprocess(image, device=device)
    return predict(model, tensor, device)


def main():
    parser = argparse.ArgumentParser(description='Facial expression recognition')
    parser.add_argument('--image', type=str, help='Path to input image')
    parser.add_argument('--image-dir', type=str, help='Batch process a directory of images')
    parser.add_argument('--cpu', action='store_true', help='Force CPU inference')

    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
    print(f"Device: {device}")

    model = load_model(device)

    if args.image:
        emotion, confidence, probs = predict_image(model, args.image, device)
        print(f"\nImage: {args.image}")
        print(f"Prediction: {emotion} (confidence: {confidence*100:.1f}%)")
        print("\nPer-class probabilities:")
        for emo, prob in sorted(probs.items(), key=lambda x: x[1], reverse=True):
            bar = '#' * int(prob * 50)
            print(f"  {emo:10s}: {prob*100:5.1f}% {bar}")

    if args.image_dir:
        print(f"\nBatch processing: {args.image_dir}")
        extensions = ('.jpg', '.jpeg', '.png', '.bmp')
        files = [f for f in os.listdir(args.image_dir) if f.lower().endswith(extensions)]
        print(f"Found {len(files)} images")

        for fname in sorted(files):
            path = os.path.join(args.image_dir, fname)
            emotion, confidence, _ = predict_image(model, path, device)
            print(f"  {fname:30s} -> {emotion:8s} ({confidence*100:.1f}%)")


if __name__ == '__main__':
    main()
