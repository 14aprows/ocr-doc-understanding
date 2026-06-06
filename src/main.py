import torch
import torch.optim as optim

from src.configs.config import (
    EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    LOG_DIR,
    BEST_MODEL_PATH,
    LAST_MODEL_PATH,
)
from src.data.ocr_dataloader import get_ocr_dataloader
from src.models.crnn import CRNN
from src.losses.ctc_loss import CRNNCTCLoss
from src.utils.text_encoder import TextEncoder
from src.utils.csv_logger import get_csv_log_path, init_csv_log, append_csv_log
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

    log_path = get_csv_log_path(LOG_DIR, model)
    init_csv_log(log_path)

    BEST_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    criterion = CRNNCTCLoss(blank_idx=encoder.blank_idx)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    best_val_loss = float("inf")

    for epoch in range(EPOCHS):
        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        val_loss, cer = validate_one_epoch(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            encoder=encoder,
        )

        checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "cer": cer,
        }

        torch.save(checkpoint, LAST_MODEL_PATH)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(checkpoint, BEST_MODEL_PATH)

        append_csv_log(
            log_path=log_path,
            epoch=epoch + 1,
            train_loss=train_loss,
            val_loss=val_loss,
            cer=cer,
        )

        print(
            f"Epoch [{epoch + 1}/{EPOCHS}] "
            f"Train Loss: {train_loss:.4f} "
            f"Val Loss: {val_loss:.4f} "
            f"CER: {cer:.4f}"
        )

if __name__ == "__main__":
    main()