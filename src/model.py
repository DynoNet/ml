import math
import torch
import torch.nn as nn

# see Mixed-Radix Number System
def encode_hold(x: torch.Tensor, y: torch.Tensor, r: torch.Tensor, vocab_y: int = 39, vocab_r: int = 6) -> torch.Tensor:
    return x * vocab_y * vocab_r + y * vocab_r + r

def decode_token(token_id: int, vocab_y: int, vocab_r: int) -> tuple[int, int, int]:
    r = token_id % vocab_r
    y = (token_id - r) // vocab_r % vocab_y
    x = (token_id - r - y * vocab_r) // (vocab_r * vocab_y)

    return x, y, r

class PositionalEncoding(nn.Module):
    # Type annotation for static analyzers
    sin_pos_enc: torch.Tensor     

    def __init__(self, d_model: int = 128, max_len: int = 41):
        super().__init__()
        positional_encodings = torch.zeros(max_len, d_model)
        sequence_step_indeces = torch.arange(start=0, end=max_len, step=1, dtype=torch.float)
        sequence_step_indeces = sequence_step_indeces.unsqueeze(1)
        #TODO: understand these sine cosine shananigans
        div_term = torch.exp(torch.arange(start=0, end=d_model, step=2, dtype=torch.float) * -math.log(10000.0, math.e) / d_model)
        
        #This is not necessary due to pytorch broadcasting
        # div_term = div_term.unsqueeze(0)
    
        #sequence_step_indeces has dimensions (max_len x 1)
        #div_term has dimensions (1 x d_model/2) after broadcasting
        #result is (1 x max_len x d_model)
        positional_encodings[:, 0::2] = torch.sin(sequence_step_indeces * div_term)
        positional_encodings[:, 1::2] = torch.cos(sequence_step_indeces * div_term)

        positional_encodings = positional_encodings.unsqueeze(0)
        self.register_buffer("sin_pos_enc", positional_encodings)
    
    def forward(self, sequence_tensor):
        #sequence_tensor : (batch_size, seq_len, d_model)
        seq_len = sequence_tensor.size(1)
        #again broadcasting at play in order for this addition to work
        return sequence_tensor + self.sin_pos_enc[:, :seq_len, :]

class Model(nn.Module):
    def __init__(self, vocab_x: int = 36, vocab_y: int = 39, vocab_r: int = 6, d_model: int = 128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_x * vocab_y * vocab_r, d_model, padding_idx=0)
        self.vocab_x = vocab_x
        self.vocab_y = vocab_y
        self.vocab_r = vocab_r
        self.total_vocab = vocab_x * vocab_y * vocab_r
        #TODO change to 3 when we implement climb types (see to do for details)
        self.meta_proj = nn.Linear(2, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(d_model, 8, 512, 0.1, "gelu", batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, 4, nn.LayerNorm(d_model))

        self.logit = nn.Linear(d_model, self.total_vocab)

    def _to_token_ids(self, holds: torch.Tensor) -> torch.Tensor:
        if holds.dim() == 3:
            x = holds[:, :, 0].long()
            y = holds[:, :, 1].long()
            r = holds[:, :, 2].long()
            return encode_hold(x, y, r, self.vocab_y, self.vocab_r)
        return holds.long()

    def forward(self, meta: torch.Tensor, holds: torch.Tensor):
        ids = self._to_token_ids(holds)
        holds_projs = self.embedding(ids)
        meta_projs = self.meta_proj(meta).unsqueeze(1)

        tokens = torch.cat([meta_projs, holds_projs], 1)

        pe_tokens = self.pos_encoder(tokens)
        
        padding_mask = ids == 0
        meta_padding_mask = torch.zeros(padding_mask.size(0), 1, dtype=torch.bool, device=padding_mask.device)
        padding_mask = torch.cat([meta_padding_mask, padding_mask], dim=1)

        causal_mask = nn.Transformer.generate_square_subsequent_mask(tokens.size(1), device=tokens.device)

        output = self.transformer(pe_tokens, mask=causal_mask, src_key_padding_mask = padding_mask, is_causal=True)

        return self.logit(output)

       
