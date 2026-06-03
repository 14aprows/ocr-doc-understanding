import torch 

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

def validate_one_epoch(model, dataloader, criterion, device):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in dataloader:
            images = batch["images"].to(device)
            targets = batch["targets"].to(device)
            target_lengths = batch["target_lengths"].to(device)
            logits = model(images)
            loss = criterion(logits, targets, target_lengths)
            val_loss += loss.item()
    return val_loss / len(dataloader)