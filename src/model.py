import math
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=41):
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

class HoldEmbedding(nn.Module):
    def __init__(self, d_model, num_x, num_y, num_r):
        super().__init__()
        self.x_lk_t = nn.Embedding(num_x, d_model)
        self.y_lk_t = nn.Embedding(num_y, d_model)
        self.r_lk_t = nn.Embedding(num_r, d_model)
    
    def forward(self, x):
        # x shape: (batch_size, seq_len, 3)
        x_emb = self.x_lk_t(x[:, :, 0].long())
        y_emb = self.y_lk_t(x[:, :, 1].long())
        r_emb = self.r_lk_t(x[:, :, 2].long())

        return x_emb + y_emb + r_emb

