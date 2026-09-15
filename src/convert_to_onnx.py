import torch
from model import Model

device = torch.device("cpu")

model = Model(vocab_x=36, vocab_y=45, vocab_r=6).to(device)
model.load_state_dict(torch.load("best_model.pt", map_location=device))
model.eval()

# Change seq_len from 1 to 5 so ONNX sees a true dynamic axis
dummy_meta = torch.randn(1, 2, dtype=torch.float32, device=device)
dummy_holds = torch.zeros((1, 5, 3), dtype=torch.long, device=device) 

torch.onnx.export(
    model,
    (dummy_meta, dummy_holds),
    "kilter_model.onnx",
    export_params=True,
    opset_version=17,
    do_constant_folding=True,  # Set to True
    input_names=["meta", "holds"],
    output_names=["logits"],
    dynamic_axes={
        "meta": {0: "batch_size"},
        "holds": {0: "batch_size", 1: "seq_len"},
        "logits": {0: "batch_size", 1: "total_seq_len"},
    },
    dynamo=False
)

print("Model exported.")
