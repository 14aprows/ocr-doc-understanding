from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset

class OCRDataset(Dataset):
    def __init__(self, images_dir, labels_path, transform=None):
        self.images_dir = Path(images_dir)
        self.labels_path = Path(labels_path)
        self.transform = transform

        self.samples = []
        with self.labels_path.open('r', encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line == "":
                    continue
                image_name, text = line.split("\t", maxsplit=1)
                image_path = self.images_dir / image_name
                if not image_path.exists():
                    print(f"Image not found: {image_name}")
                    continue
                self.samples.append({
                    "image_path": image_path,
                    "text": text
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample["image_path"]).convert("L")
        if self.transform:
            image = self.transform(image)
        return {
            "image": image,
            "text": sample["text"],
            "image_path": str(sample["image_path"])
        }