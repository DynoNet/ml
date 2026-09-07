import torch
from model import Model

max_sequence_length = 30
temperature = 0.8

@torch.no_grad()
def generate_climb(model: Model, angle: float, grade: float, device: torch.device | str = "cpu"):
    model.eval()

    meta = torch.tensor([[angle, grade]], dtype=torch.float32, device=device)
    holds = torch.empty((1, 0, 3), dtype=torch.long, device=device)

    used_coords = set()
    current_max_y = 0

    for i in range(max_sequence_length):
        x_hat, y_hat, r_hat = model(meta, holds)
        
        x_logits = x_hat[0, -1, :].clone()
        y_logits = y_hat[0, -1, :].clone()
        r_logits = r_hat[0, -1, :].clone()

        # Prevent downward movement (allow same Y or 1 row lower for foot holds)
        if current_max_y > 2:
            y_logits[: max(0, current_max_y - 1)] = float("-inf")

        # Role constraints
        if i < 2:
            # Force start holds on initial steps
            r_logits[3] = float("-inf")
            r_logits[4] = float("-inf")
            r_logits[5] = float("-inf")
        else:
            # Block start holds after step 2
            r_logits[2] = float("-inf")

        if i < 4:
            # Prevent finish holds before step 4
            r_logits[4] = float("-inf")

        p_x = torch.softmax(x_logits / temperature, dim=-1)
        p_y = torch.softmax(y_logits / temperature, dim=-1)
        p_r = torch.softmax(r_logits / temperature, dim=-1)

        # Rejection sampling for exact (X, Y) duplicates only
        while True:
            next_x = torch.multinomial(p_x, 1).item()
            next_y = torch.multinomial(p_y, 1).item()
            if (next_x, next_y) not in used_coords:
                break

        next_r = torch.multinomial(p_r, 1).item()

        # Force finish role if reaching the top row (Y >= 37) after enough holds
        if next_y >= 37 and i >= 5:
            next_r = 4

        used_coords.add((next_x, next_y))
        if next_y > current_max_y:
            current_max_y = next_y

        next_hold = torch.tensor([[[next_x, next_y, next_r]]], dtype=torch.long, device=device)
        holds = torch.cat([holds, next_hold], dim=1)

        if next_r == 4:
            break

    return holds.squeeze(0).tolist()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Model(vocab_x=36, vocab_y=45, vocab_r=6).to(device)
    model.load_state_dict(torch.load("best_model.pt", map_location=device))

    angle = 45
    grade = 17
    climb = generate_climb(model, angle, grade, device)

    print(f"\nGenerated Climb (Angle: {angle}°, Grade: {grade})")
    print("-" * 30)
    print(f"{'Hold':<6} | {'X':<6} | {'Y':<6} | {'Role':<6}")
    print("-" * 30)
    for idx, (x, y, r) in enumerate(climb, 1):
        print(f"{idx:<6} | {x:<6} | {y:<6} | {r:<6}")
    print("-" * 30)
    print(f"Total holds: {len(climb)}\n")
