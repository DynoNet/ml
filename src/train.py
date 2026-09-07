import dataloader
from model import Model
import torch
import time
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

def train_one_epoch(train_loader: DataLoader, model:Model, optimizer, device):
    running_loss = 0

    # sets dropout and batch norm
    model.train()
    
    #loss computation
    loss_criterion = nn.CrossEntropyLoss(ignore_index = 0)

    for batch in train_loader:

        input_seq = batch["input_seq"].to(device)
        target_seq = batch["target_seq"].to(device)

        #unpack the meta from the sequence
        meta_batch = input_seq[:, 0, :2]
        hold_batch = input_seq[:, 1:, :]

        #pass input_seq to the model
        x_hat, y_hat, r_hat = model(meta_batch, hold_batch)

        #recieve 3 heads and compare to target_seq
        x = target_seq[:, :, 0]
        y = target_seq[:, :, 1]
        r = target_seq[:, :, 2]

        x_hat = x_hat.reshape(-1, x_hat.size(-1))
        y_hat = y_hat.reshape(-1, y_hat.size(-1))
        r_hat = r_hat.reshape(-1, r_hat.size(-1))

        x = x.reshape(-1)
        y = y.reshape(-1)
        r = r.reshape(-1)


        total_loss = loss_criterion(x_hat, x) + loss_criterion(y_hat, y) + loss_criterion(r_hat, r)
        running_loss = running_loss + total_loss.item()
        #backwards pass & optimizer
        #optimizer
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    return running_loss / len(train_loader)

    
def validate(val_loader: DataLoader, model:Model, device):    
    running_loss = 0

    # sets dropout and batch norm
    model.eval()
    
    #loss computation
    loss_criterion = nn.CrossEntropyLoss(ignore_index = 0)
    with torch.no_grad():
        for batch in val_loader:

                input_seq = batch["input_seq"].to(device)
                target_seq = batch["target_seq"].to(device)

                #unpack the meta from the sequence
                meta_batch = input_seq[:, 0, :2]
                hold_batch = input_seq[:, 1:, :]

                #pass input_seq to the model
                x_hat, y_hat, r_hat = model(meta_batch, hold_batch)

                #recieve 3 heads and compare to target_seq
                x = target_seq[:, :, 0]
                y = target_seq[:, :, 1]
                r = target_seq[:, :, 2]

                x_hat = x_hat.reshape(-1, x_hat.size(-1))
                y_hat = y_hat.reshape(-1, y_hat.size(-1))
                r_hat = r_hat.reshape(-1, r_hat.size(-1))

                x = x.reshape(-1)
                y = y.reshape(-1)
                r = r.reshape(-1)


                total_loss = loss_criterion(x_hat, x) + loss_criterion(y_hat, y) + loss_criterion(r_hat, r)
                running_loss = running_loss + total_loss.item()

    return running_loss / len(val_loader)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ldr, validate_ldr, test_ldr = dataloader.get_dataloaders("/home/tudor/Code/DynoNet/data/processed/dataset.pt") 
    
    # all_input = train_ldr.dataset[:]["input_seq"]
    # max_x = all_input[:, :, 0].max().item()
    # max_y = all_input[:, :, 1].max().item()
    # max_r = all_input[:, :, 2].max().item()
    #
    # print(f"Max X: {max_x}, Max Y: {max_y}, Max Role: {max_r}")
    
    # Max X: 35.0, Max Y: 38.0, Max Role: 5.0

    #TODO change this to 39 and train again
    model = Model(36, 45, 6).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    best_val_loss = float("inf")
    num_epochs = 20

    print(f"Training on device: {device}\n")

    for epoch in range(num_epochs):
        start_time = time.time()

        train_loss = train_one_epoch(train_ldr, model, optimizer, device)
        val_loss = validate(validate_ldr, model, device)

        epoch_time = time.time() - start_time
        current_lr = optimizer.param_groups[0]["lr"]

        # Checkpoint if validation loss improves
        saved_tag = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pt")
            saved_tag = " -> [Saved Best Model]"

        print(
            f"Epoch {epoch + 1:02d}/{num_epochs:02d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"LR: {current_lr:.1e} | "
            f"Time: {epoch_time:.2f}s"
            f"{saved_tag}"
        )
