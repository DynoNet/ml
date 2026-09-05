import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class KilterTransformer(nn.Module):

    def __init__(
        self,
        vocab_x=36,
        vocab_y=45,
        vocab_role=6,
        d_model=128,
        nhead=8,
        num_layers=4,
    ):
        super().__init__()

        # 1. Discrete feature embeddings for holds
        self.emb_x = nn.Embedding(vocab_x, 32)
        self.emb_y = nn.Embedding(vocab_y, 64)
        self.emb_role = nn.Embedding(vocab_role, 32)
        self.hold_proj = nn.Linear(32 + 64 + 32, d_model)

        # 2. Linear projection for metadata [angle, grade]
        self.meta_proj = nn.Linear(2, d_model)

        # 3. Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)

        # 4. Transformer Decoder Stack
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=512, batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=num_layers
        )

        # 5. Output Classification Heads
        self.head_x = nn.Linear(d_model, vocab_x)
        self.head_y = nn.Linear(d_model, vocab_y)
        self.head_role = nn.Linear(d_model, vocab_role)

    def forward(self, input_seq, padding_mask):
        # input_seq: [batch_size, seq_len, 3]
        # padding_mask: [batch_size, seq_len]

        batch_size, seq_len, _ = input_seq.shape

        # 1. Separate metadata (token 0) and hold sequence (tokens 1..N-1)
        meta_inputs = input_seq[:, 0, :2]  # [batch_size, 2]
        hold_inputs = input_seq[
            :, 1:, :
        ].long()  # [batch_size, seq_len - 1, 3]

        # 2. Project metadata to d_model space
        meta_emb = self.meta_proj(meta_inputs).unsqueeze(
            1
        )  # [batch_size, 1, d_model]

        # 3. Look up embeddings for holds and project to d_model space
        x_e = self.emb_x(
            hold_inputs[:, :, 0]
        )  # [batch_size, seq_len - 1, 32]
        y_e = self.emb_y(
            hold_inputs[:, :, 1]
        )  # [batch_size, seq_len - 1, 64]
        r_e = self.emb_role(
            hold_inputs[:, :, 2]
        )  # [batch_size, seq_len - 1, 32]

        hold_emb = self.hold_proj(
            torch.cat([x_e, y_e, r_e], dim=-1)
        )  # [batch_size, seq_len - 1, d_model]

        # 4. Re-combine sequence along time dimension (dim=1)
        x = torch.cat(
            [meta_emb, hold_emb], dim=1
        )  # [batch_size, seq_len, d_model]

        # 5. Add positional encodings
        x = self.pos_encoder(x)

        # 6. Generate causal mask (upper triangular matrix of -inf)
        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=x.device),
            diagonal=1,
        )

        # 7. Pass through Transformer Decoder
        out = self.transformer_decoder(
            tgt=x,
            memory=x,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=padding_mask,
        )

        # 8. Calculate output logits for next X, Y, and Role
        logits_x = self.head_x(out)  # [batch_size, seq_len, vocab_x]
        logits_y = self.head_y(out)  # [batch_size, seq_len, vocab_y]
        logits_role = self.head_role(
            out
        )  # [batch_size, seq_len, vocab_role]

        return logits_x, logits_y, logits_role

