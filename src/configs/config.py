from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "dataset"
TRAIN_DIR = DATA_DIR / "training_data"
VAL_DIR = DATA_DIR / "testing_data"

TRAIN_IMAGES_DIR = TRAIN_DIR / "images"
VAL_IMAGES_DIR = VAL_DIR / "images"

TRAIN_ANNOTATIONS_DIR = TRAIN_DIR / "annotations"
VAL_ANNOTATIONS_DIR = VAL_DIR / "annotations"

IMAGE_SIZE = 224
BATCH_SIZE = 2
EPOCHS = 10
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 0.01
SEED=42

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"