from pathlib import Path
# import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "dataset"
TRAIN_DIR = DATA_DIR / "training_data"
VAL_DIR = DATA_DIR / "testing_data"

TRAIN_IMAGES_DIR = TRAIN_DIR / "images"
VAL_IMAGES_DIR = VAL_DIR / "images"

TRAIN_ANNOTATIONS_DIR = TRAIN_DIR / "annotations"
VAL_ANNOTATIONS_DIR = VAL_DIR / "annotations"

OCR_DATA_DIR = PROJECT_ROOT / "dataset_ocr"
OCR_TRAIN_DIR = OCR_DATA_DIR / "train"
OCR_VAL_DIR = OCR_DATA_DIR / "val"

OCR_TRAIN_IMAGES_DIR = OCR_TRAIN_DIR / "images"
OCR_VAL_IMAGES_DIR = OCR_VAL_DIR / "images"
OCR_TRAIN_LABELS_PATH = OCR_TRAIN_DIR / "labels.txt"
OCR_VAL_LABELS_PATH = OCR_VAL_DIR / "labels.txt"

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
BEST_MODEL_PATH = CHECKPOINT_DIR / "best_ocr_model.pth"
LAST_MODEL_PATH = CHECKPOINT_DIR / "last_ocr_model.pth"

LOG_DIR = PROJECT_ROOT / "logs"
TRAIN_LOG_PATH = LOG_DIR / "ocr_training_log.csv"

OCR_IMAGE_HEIGHT = 32
OCR_IMAGE_WIDTH = 128

IMAGE_SIZE = 224
BATCH_SIZE = 2
EPOCHS = 10
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 0.01
SEED=42

# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"