from __future__ import annotations

from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def train_transform(image_size: int = 224):
    """Conservative augmentation for forensic image classification."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.08, contrast=0.08, saturation=0.06, hue=0.02),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def eval_transform(image_size: int = 224):
    """Deterministic transform for validation/test/inference."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
