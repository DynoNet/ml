import torch

ROLE_NAMES = {1: "Meta", 2: "Start", 3: "Middle", 4: "Finish", 5: "Foot"}

data = torch.load("/home/tudor/Code/DynoNet/data/processed/dataset.pt")

# Inspect the first 5 climbs
for i in range(5):
    climb = data[i]
    meta = climb[0]
    print(f"=== Climb {i + 1} | Angle: {meta[0]}, Grade: {meta[1]} ===")
    
    for token in climb[1:]:
        x, y, r = int(token[0]), int(token[1]), int(token[2])
        if r == 0:  # Skip padding
            continue
        print(f"  X: {x:2d} | Y: {y:2d} | Role: {ROLE_NAMES.get(r, r)}")
    print()
