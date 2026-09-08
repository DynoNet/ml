import torch
import dataloader
from model import Model, decode_token

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = Model(vocab_x=36, vocab_y=45, vocab_r=6).to(device)
model.load_state_dict(torch.load("best_model.pt", map_location=device))
model.eval()

# Load 1 batch from validation set
_, val_ldr, _ = dataloader.get_dataloaders("/home/tudor/Code/DynoNet/data/processed/dataset.pt")
batch = next(iter(val_ldr))

input_seq = batch["input_seq"].to(device)
meta = input_seq[0:1, 0, :2]
real_holds = input_seq[0:1, 1:, :]

print("==================================================")
print("1. META FEATURE VALUES FROM DATALOADER")
print("==================================================")
print(f"Meta tensor: {meta.squeeze(0).tolist()}\n")

print("==================================================")
print("2. STEP 0 TOP-5 PREDICTIONS (GIVEN ONLY META)")
print("==================================================")
empty_holds = torch.empty((1, 0, 3), dtype=torch.long, device=device)
with torch.no_grad():
    logits_step0 = model(meta, empty_holds)[0, -1, :]
    top5_idx = torch.topk(logits_step0, 5).indices.tolist()
    for rank, tok in enumerate(top5_idx, 1):
        x, y, r = decode_token(tok, model.vocab_y, model.vocab_r)
        prob = torch.softmax(logits_step0, dim=-1)[tok].item()
        print(f"Rank {rank}: X={x:<2} | Y={y:<2} | Role={r} | Prob={prob:.4f}")

print("\n==================================================")
print("3. TEACHER FORCING CHECK ON REAL CLIMB")
print("==================================================")
# Test prediction after feeding 3 real holds
prefix_len = 3
prefix_holds = real_holds[:, :prefix_len, :]

with torch.no_grad():
    logits_prefix = model(meta, prefix_holds)[0, -1, :]
    top3_idx = torch.topk(logits_prefix, 3).indices.tolist()

print("Real Prefix Holds:")
for idx, h in enumerate(prefix_holds.squeeze(0).tolist(), 1):
    print(f"  Hold {idx}: X={h[0]}, Y={h[1]}, Role={h[2]}")

real_next = real_holds[0, prefix_len, :].tolist()
print(f"\nGround Truth Hold {prefix_len + 1}: X={real_next[0]}, Y={real_next[1]}, Role={real_next[2]}")
print("Model Top-3 Predictions for Next Hold:")
for rank, tok in enumerate(top3_idx, 1):
    x, y, r = decode_token(tok, model.vocab_y, model.vocab_r)
    prob = torch.softmax(logits_prefix, dim=-1)[tok].item()
    print(f"  Rank {rank}: X={x:<2} | Y={y:<2} | Role={r} | Prob={prob:.4f}")
