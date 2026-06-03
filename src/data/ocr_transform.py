from torchvision import transforms
from src.configs.config import OCR_IMAGE_HEIGHT, OCR_IMAGE_WIDTH

def get_ocr_transform():
    return transforms.Compose([
        transforms.Resize((OCR_IMAGE_HEIGHT, OCR_IMAGE_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])