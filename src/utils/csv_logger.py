import csv

def get_model_name(class_model):
    if isinstance(class_model, str):
        return class_model.lower()
    return class_model.__class__.__name__.lower()

def get_csv_log_path(log_dir, class_model):
    model_name = get_model_name(class_model)
    return log_dir / f"{model_name}_training_log.csv"

def init_csv_log(log_path):
    log_path.parent.mkdir(parents=True, exist_ok=True)

def append_csv_log(log_path, epoch, train_loss, val_loss, cer):
    with log_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([epoch, train_loss, val_loss, cer])