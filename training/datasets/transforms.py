"""Image transforms for MedVision AI chest X-ray training and inference."""

from torchvision import transforms

# ImageNet normalization constants (appropriate for DenseNet121 pretrained weights)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Target input size for DenseNet121
INPUT_SIZE = 224


def get_train_transform(image_size: int = INPUT_SIZE) -> transforms.Compose:
    """
    Training augmentation pipeline.
    Conservative augmentations appropriate for medical imaging:
    - No aggressive colour jitter (radiographs are grayscale)
    - No vertical flip (anatomical orientation must be preserved)
    - Mild horizontal flip only (AP/PA views may allow this)
    - Rotation limited to ±7 degrees
    """
    return transforms.Compose([
        transforms.Resize((image_size + 32, image_size + 32)),
        transforms.RandomCrop(image_size),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.RandomRotation(degrees=7),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_eval_transform(image_size: int = INPUT_SIZE) -> transforms.Compose:
    """
    Evaluation/inference transform — deterministic, no augmentation.
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_gradcam_transform() -> transforms.Compose:
    """
    Transform for Grad-CAM: same as eval but returns 224x224 tensors.
    """
    return get_eval_transform()
