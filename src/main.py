import torch
import torch.optim as optim

from src.configs.config import (
    EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
)
from src.data.ocr_dataloader import get_ocr_dataloader
from src.models.crnn import CRNN
from src.losses.ctc_loss import CRNNCTCLoss
from src.utils.text_encoder import TextEncoder
from src.engine.trainer import train_one_epoch, validate_one_epoch

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    train_loader, val_loader = get_ocr_dataloader()

    encoder = TextEncoder()

    model = CRNN(
        in_channels=1,
        num_classes=encoder.num_classes(),
    ).to(device)

    criterion = CRNNCTCLoss(blank_idx=encoder.blank_idx)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    for epoch in range(EPOCHS):
        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        val_loss = validate_one_epoch(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
        )

        print(
            f"Epoch [{epoch + 1}/{EPOCHS}] "
            f"Train Loss: {train_loss:.4f} "
            f"Val Loss: {val_loss:.4f}"
        )

if __name__ == "__main__":
    main()