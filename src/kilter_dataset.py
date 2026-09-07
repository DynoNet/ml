import json
import torch
from torch.utils.data import Dataset


class KilterDataset(Dataset):

    def __init__(self, data_path, max_angle=70.0, max_diff=39.0):
        if data_path.endswith(".pt"):
            raw_data = torch.load(data_path)
        else:
            return
        self.sequences = torch.tensor(raw_data, dtype=torch.float32)
        self.sequences[:, 0, 0] /= max_angle
        self.sequences[:, 0, 1] /= max_diff

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]

        # 2. Autoregressive target shift
        #TODO Read more about this one
        input_seq = seq[:-1]  # Tokens 0 to N-1
        target_seq = seq[1:]  # Tokens 1 to N

        return {
            "input_seq": input_seq,
            "target_seq": target_seq.long(),
        }

