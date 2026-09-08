import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import dataloader
from model import Model, encode_hold


def train_one_epoch(train_loader: DataLoader, model: Model, optimizer, loss_criterion, device):
    running_loss = 0.0
    model.train()

    for batch in train_loader:
        input_seq = batch["input_seq"].to(device)
        target_seq = batch["target_seq"].to(device)

        meta_batch = input_seq[:, 0, :2]
        hold_batch = input_seq[:, 1:, :]

        logits = model(meta_batch, hold_batch)

        x_target = target_seq[:, :, 0]
        y_target = target_seq[:, :, 1]
        r_target = target_seq[:, :, 2]
        target_tokens = encode_hold(x_target, y_target, r_target, model.vocab_y, model.vocab_r)

        # No slicing. logits and target_tokens are already perfectly aligned (B, seq_len)
        loss = loss_criterion(
            logits.reshape(-1, model.vocab_size),
            target_tokens.reshape(-1)
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(train_loader)


@torch.no_grad()
def validate(val_loader: DataLoader, model: Model, loss_criterion, device):
    running_loss = 0.0
    model.eval()

    for batch in val_loader:
        input_seq = batch["input_seq"].to(device)
        target_seq = batch["target_seq"].to(device)

        meta_batch = input_seq[:, 0, :2]
        hold_batch = input_seq[:, 1:, :]

        logits = model(meta_batch, hold_batch)

        x_target = target_seq[:, :, 0]
        y_target = target_seq[:, :, 1]
        r_target = target_seq[:, :, 2]
        target_tokens = encode_hold(x_target, y_target, r_target, model.vocab_y, model.vocab_r)

        loss = loss_criterion(
            logits.reshape(-1, model.vocab_size),
            target_tokens.reshape(-1)
        )
        running_loss += loss.item()

    return running_loss / len(val_loader)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ldr, validate_ldr, _ = dataloader.get_dataloaders("/home/tudor/Code/DynoNet/data/processed/dataset.pt")

    num_epochs = 60
    initial_lr = 1e-3
    min_lr = 1e-6

    model = Model(vocab_x=36, vocab_y=45, vocab_r=6).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=initial_lr, weight_decay=1e-2)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=min_lr
    )

    loss_criterion = nn.CrossEntropyLoss(ignore_index=0)
    best_val_loss = float("inf")

    print(f"Training on device: {device} for {num_epochs} epochs\n")

    for epoch in range(num_epochs):
        start_time = time.time()

        train_loss = train_one_epoch(train_ldr, model, optimizer, loss_criterion, device)
        val_loss = validate(validate_ldr, model, loss_criterion, device)

        scheduler.step()

        epoch_time = time.time() - start_time
        current_lr = optimizer.param_groups[0]["lr"]

        saved_tag = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pt")
            saved_tag = " -> [Saved Best Model]"

        print(
            f"Epoch {epoch + 1:02d}/{num_epochs:02d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"LR: {current_lr:.2e} | "
            f"Time: {epoch_time:.2f}s"
            f"{saved_tag}"
        )
