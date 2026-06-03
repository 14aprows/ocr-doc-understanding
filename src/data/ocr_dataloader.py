from torch.utils.data import DataLoader
from src.configs.config import BATCH_SIZE, OCR_TRAIN_IMAGES_DIR, OCR_TRAIN_DIR, OCR_VAL_IMAGES_DIR, OCR_VAL_LABELS_DIR
from src.data.ocr_dataset import OCRDataset
from src.data.ocr_transform import get_ocr_transform
from src.utils.collate import collate_fn

def get_ocr_dataloader():
    transform = get_ocr_transform()

    train_dataset = OCRDataset(
        images_dir = OCR_TRAIN_IMAGES_DIR,
        labels_path = OCR_TRAIN_DIR,
        transform = transform
    )

    val_dataset = OCRDataset(
        images_dir = OCR_VAL_IMAGES_DIR,
        labels_path = OCR_VAL_LABELS_DIR,
        transform = transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
    )

    return train_loader, val_loader