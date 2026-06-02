from torch.utils.data import DataLoader
from configs.config import BATCH_SIZE, TRAIN_ANNOTATIONS_DIR, TRAIN_IMAGES_DIR, VAL_ANNOTATIONS_DIR, VAL_IMAGES_DIR
from data.dataset import DocumentDataset

def collate_fn(batch):
    return {
        "images": [item["image"] for item in batch],
        "image_paths": [item["image_path"] for item in batch],
        "words": [item["words"] for item in batch],
        "boxes": [item["boxes"] for item in batch],
        "labels": [item["labels"] for item in batch],
        "label_names": [item["label_names"] for item in batch]
    }

def get_dataloader():
    train_dataset = DocumentDataset(
        image_dir = TRAIN_IMAGES_DIR,
        annotations_dir = TRAIN_ANNOTATIONS_DIR
    )

    val_dataset = DocumentDataset(
        image_dir = VAL_IMAGES_DIR,
        annotations_dir = VAL_ANNOTATIONS_DIR
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn
    )

    return train_loader, val_loader