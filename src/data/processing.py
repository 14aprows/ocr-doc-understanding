import json
from pathlib import Path
from PIL import Image

def normalize_bbox(box, width, height):
    x0, y0, x1, y1 = box
    return [
        int(1000 * x0 / width),
        int(1000 * y0 / height),
        int(1000 * x1 / width),
        int(1000 * y1 / height)
    ]

def read_annotations(annotation_path, image_path):
    annotation_path = Path(annotation_path)
    image_path = Path(image_path)

    with annotation_path.open("r", encoding="utf-8") as f:
        annotation = json.load(f)

    image = Image.open(image_path).convert("RGB")
    width, height = image.size

    words = []
    boxes = []
    labels = []

    for item in annotation.get("form", []):
        label_name = item.get("label", "other")
        for word in item.get("words", []):
            text = word.get("text", "").strip()
            if text == "":
                continue
            box = word.get("box")
            if box is None:
                continue

            words.append(text)
            boxes.append(normalize_bbox(box, width, height))
            labels.append(label_name)
            
    return words, boxes, labels

