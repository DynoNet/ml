import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix

import dataloader
from model import Model


@torch.no_grad()
def evaluate_role_predictions(model, val_loader, device):
    model.eval()

    all_preds = []
    all_targets = []
    all_entropies = []

    for batch in val_loader:
        input_seq = batch["input_seq"].to(device)
        target_seq = batch["target_seq"].to(device)

        # Unpack metadata and holds
        meta_batch = input_seq[:, 0, :2]
        hold_batch = input_seq[:, 1:, :]

        # Forward pass
        _, _, r_hat = model(meta_batch, hold_batch)

        r_logits = r_hat.reshape(-1, r_hat.size(-1))
        r_targets = target_seq[:, :, 2].reshape(-1)

        # Filter out padding tokens (ignore_index = 0)
        mask = r_targets != 0
        r_logits = r_logits[mask]
        r_targets = r_targets[mask]

        probs = F.softmax(r_logits, dim=-1)
        log_probs = F.log_softmax(r_logits, dim=-1)
        entropy = -torch.sum(probs * log_probs, dim=-1)

        preds = torch.argmax(r_logits, dim=-1)

        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(r_targets.cpu().numpy())
        all_entropies.extend(entropy.cpu().numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_entropies = np.array(all_entropies)

    print("=" * 60)
    print(" 1. DATASET CLASS DISTRIBUTION (GROUND TRUTH - EXCL. PADDING)")
    print("=" * 60)
    unique_classes, counts = np.unique(all_targets, return_counts=True)
    total_samples = len(all_targets)
    for cls, count in zip(unique_classes, counts):
        percentage = (count / total_samples) * 100
        print(f"Role {cls:<2} : {count:<8} samples ({percentage:.2f}%)")

    print("\n" + "=" * 60)
    print(" 2. PREDICTION ENTROPY")
    print("=" * 60)
    print(f"Mean Entropy : {all_entropies.mean():.4f} (Near 0 = overconfident)")
    print(f"Min Entropy  : {all_entropies.min():.4f}")
    print(f"Max Entropy  : {all_entropies.max():.4f}")

    print("\n" + "=" * 60)
    print(" 3. CONFUSION MATRIX (Rows: True Role, Cols: Predicted Role)")
    print("=" * 60)
    cm = confusion_matrix(all_targets, all_preds)
    print(cm)

    print("\n" + "=" * 60)
    print(" 4. CLASSIFICATION REPORT (PRECISION, RECALL, F1)")
    print("=" * 60)
    print(classification_report(all_targets, all_preds, digits=4, zero_division=0))


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_path = "/home/tudor/Code/DynoNet/data/processed/dataset.pt"
    _, val_loader, _ = dataloader.get_dataloaders(data_path)

    model = Model(vocab_x=36, vocab_y=45, vocab_r=6).to(device)
    model.load_state_dict(torch.load("best_model.pt", map_location=device))

    evaluate_role_predictions(model, val_loader, device)
