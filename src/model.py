import math
import torch
import torch.nn as nn


def encode_hold(x: torch.Tensor, y: torch.Tensor, r: torch.Tensor, vocab_y: int = 45, vocab_r: int = 6) -> torch.Tensor:
    """Maps (X, Y, Role) integer tensors to a single unified token ID."""
    return x * (vocab_y * vocab_r) + y * vocab_r + r


def decode_token(token_id: int, vocab_y: int = 45, vocab_r: int = 6) -> tuple[int, int, int]:
    """Decodes a single unified token ID back into (X, Y, Role)."""
    r = token_id % vocab_r
    temp = token_id // vocab_r
    y = temp % vocab_y
    x = temp // vocab_y
    return x, y, r


class PositionalEncoding(nn.Module):
    sin_pos_enc: torch.Tensor

    def __init__(self, d_model: int, max_len: int = 41):
        super().__init__()
        positional_encodings = torch.zeros(max_len, d_model)
        sequence_step_indices = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * -math.log(10000.0) / d_model
        )

        positional_encodings[:, 0::2] = torch.sin(sequence_step_indices * div_term)
        positional_encodings[:, 1::2] = torch.cos(sequence_step_indices * div_term)

        positional_encodings = positional_encodings.unsqueeze(0)
        self.register_buffer("sin_pos_enc", positional_encodings)

    def forward(self, sequence_tensor: torch.Tensor) -> torch.Tensor:
        seq_len = sequence_tensor.size(1)
        return sequence_tensor + self.sin_pos_enc[:, :seq_len, :]


class Model(nn.Module):
    def __init__(
        self,
        vocab_x: int = 36,
        vocab_y: int = 45,
        vocab_r: int = 6,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
    ):
        super().__init__()
        self.vocab_x = vocab_x
        self.vocab_y = vocab_y
        self.vocab_r = vocab_r
        self.vocab_size = vocab_x * vocab_y * vocab_r  # 36 * 45 * 6 = 9720

        # Unified single-token embedding & output projection
        self.embedding = nn.Embedding(self.vocab_size, d_model, padding_idx=0)
        self.meta_proj = nn.Linear(2, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=512,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Single projection head over the full unified vocabulary
        self.logits_head = nn.Linear(d_model, self.vocab_size)

    def _to_token_ids(self, holds: torch.Tensor) -> torch.Tensor:
        """Converts raw holds tensor (batch, seq, 3) to token IDs (batch, seq)."""
        if holds.dim() == 3:
            x = holds[:, :, 0].long()
            y = holds[:, :, 1].long()
            r = holds[:, :, 2].long()
            return encode_hold(x, y, r, self.vocab_y, self.vocab_r)
        return holds.long()

    def forward(self, meta: torch.Tensor, holds: torch.Tensor) -> torch.Tensor:
        # 1. Map input holds to unified token IDs
        token_ids = self._to_token_ids(holds)

        # 2. Key padding mask (Token 0 represents padding)
        padding_mask = token_ids == 0
        meta_mask = torch.zeros(
            padding_mask.size(0), 1, dtype=torch.bool, device=padding_mask.device
        )
        padding_mask = torch.cat([meta_mask, padding_mask], dim=1)

        # 3. Embeddings & Positional Encoding
        hold_emb = self.embedding(token_ids)
        meta_emb = self.meta_proj(meta).unsqueeze(1)

        complete_tensor = torch.cat([meta_emb, hold_emb], dim=1)
        complete_tensor = self.pos_encoder(complete_tensor)

        # 4. Causal Masking
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            complete_tensor.size(1), device=complete_tensor.device
        )

        # 5. Transformer Pass
        out = self.transformer(
            complete_tensor,
            mask=causal_mask,
            src_key_padding_mask=padding_mask,
            is_causal=True,
        )

        # 6. Joint prediction logits over unified vocabulary (batch, seq_len + 1, vocab_size)
        return self.logits_head(out)
