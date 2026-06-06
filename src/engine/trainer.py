import torch 
from src.metrics.cer import average_cer

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    train_loss = 0.0
    for batch in dataloader:
        images = batch["images"].to(device)
        targets = batch["targets"].to(device)
        target_lengths = batch["target_lengths"].to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, targets, target_lengths)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    return train_loss / len(dataloader)

def validate_one_epoch(model, dataloader, criterion, device, encoder=None):
    model.eval()
    val_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in dataloader:
            images = batch["images"].to(device)
            targets = batch["targets"].to(device)
            target_lengths = batch["target_lengths"].to(device)
            logits = model(images)
            loss = criterion(logits, targets, target_lengths)
            val_loss += loss.item()

            if encoder is not None:
                pred_indices = logits.argmax(dim=2)
                for i in range(pred_indices.size(1)):
                    pred = pred_indices[:, i]
                    pred_text = encoder.decode(pred)
                    all_preds.append(pred_text)
                all_targets.extend(batch["texts"])

    avg_loss = val_loss / len(dataloader)
    if encoder is None:
        return avg_loss

    cer = average_cer(all_preds, all_targets)
    return avg_loss, cer