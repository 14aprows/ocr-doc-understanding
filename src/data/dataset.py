from cProfile import label
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from data.processing import read_annotations

LABEL2ID = {
    "other": 0,
    "question": 1,
    "answer": 2,
    "header": 3,
}

ID2LABEL = {
    0: "other",
    1: "question",
    2: "answer",
    3: "header",
}

class DocumentDataset(Dataset):
    def __init__(self, images_dir, annotations_dir):
        self.images_dir = Path(images_dir)
        self.annotations_dir = Path(annotations_dir)
        self.annotation_paths = sorted(self.annotations_dir.glob("*.json"))
        self.samples = []

        for annotation_path in self.annotation_paths:
            image_path = self.images_dir / f"{annotation_path.stem}.png"
            if image_path.exists():
                self.samples.append({
                    "image_path": image_path,
                    "annotation_path": annotation_path
                })
            else:
                print(f"Image not found: {image_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample["image_path"]).convert("RGB")

        word, boxes, labels = read_annotation(
            annotation_path = sample["annotation_path"],
            image_path = sample["image_path"]
        )

        label_ids = [LABEL2ID[label] for label in labels]

        return {
            "image": image,
            "image_path": str(sample["image_path"]),
            "words": word,
            "boxes": boxes,
            "labels": label_ids,
            "label_names": labels,
        }
