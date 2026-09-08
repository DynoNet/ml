import torch
from model import Model, decode_token

max_sequence_length = 30
temperature = 0.8

def normalize_meta(angle: float, grade: float, min_angle=0.0, max_angle=70.0, min_grade=10.0, max_grade=39.0):
    """Normalize angle and grade to match dataloader scaling [0, 1]."""
    norm_angle = (angle - min_angle) / (max_angle - min_angle)
    norm_grade = (grade - min_grade) / (max_grade - min_grade)
    return norm_angle, norm_grade

@torch.no_grad()
def generate_climb(model: Model, angle: float, grade: float, device: torch.device | str = "cpu", temp: float = 0.8):
    model.eval()

    # Normalize meta inputs before passing to model
    norm_angle, norm_grade = normalize_meta(angle, grade)
    meta = torch.tensor([[norm_angle, norm_grade]], dtype=torch.float32, device=device)
    holds = torch.empty((1, 0, 3), dtype=torch.long, device=device)

    for _ in range(max_sequence_length):
        logits = model(meta, holds)
        next_logits = logits[0, -1, :]

        if temp <= 0.0:
            next_token_id = torch.argmax(next_logits, dim=-1).item()
        else:
            probs = torch.softmax(next_logits / temp, dim=-1)
            next_token_id = torch.multinomial(probs, 1).item()

        next_x, next_y, next_r = decode_token(next_token_id, model.vocab_y, model.vocab_r)

        next_hold = torch.tensor([[[next_x, next_y, next_r]]], dtype=torch.long, device=device)
        holds = torch.cat([holds, next_hold], dim=1)

        # Stop when model predicts Finish hold (Role 4)
        if next_r == 4:
            break

    return holds.squeeze(0).tolist()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Model(vocab_x=36, vocab_y=45, vocab_r=6).to(device)
    model.load_state_dict(torch.load("best_model.pt", map_location=device))

    angle = 45.0
    grade = 17.0
    climb = generate_climb(model, angle, grade, device, temp=temperature)

    print(f"\nGenerated Climb (Angle: {angle}°, Grade: {grade})")
    print("-" * 30)
    print(f"{'Hold':<6} | {'X':<6} | {'Y':<6} | {'Role':<6}")
    print("-" * 30)
    for idx, (x, y, r) in enumerate(climb, 1):
        print(f"{idx:<6} | {x:<6} | {y:<6} | {r:<6}")
    print("-" * 30)
    print(f"Total holds: {len(climb)}\n")
