import torch 
from src.utils.text_encoder import TextEncoder

def collate_fn(batch):
    encoder = TextEncoder()
    images = [item["image"] for item in batch]
    texts = [item["text"] for item in batch]
    image_paths = [item["image_path"] for item in batch]

    images = torch.stack(images, dim=0)
    targets = []
    target_lengths = []
    for text in texts:
        encoded = encoder.encode(text)
        if len(encoded) == 0:
            continue
        targets.extend(encoded)
        target_lengths.append(len(encoded))

    targets = torch.tensor(targets, dtype=torch.long)
    target_lengths = torch.tensor(target_lengths, dtype=torch.long)

    return {
        "images": images,
        "texts": texts,
        "targets": targets,
        "target_lengths": target_lengths,
        "image_paths": image_paths,     
    }