from torch.utils.data import DataLoader, random_split
from kilter_dataset import KilterDataset


def get_dataloaders(
    data_path, batch_size=64, train_ratio=0.85, val_ratio=0.10
):
  dataset = KilterDataset(data_path)

  total_count = len(dataset)
  train_size = int(train_ratio * total_count)
  val_size = int(val_ratio * total_count)
  test_size = total_count - train_size - val_size

  train_set, val_set, test_set = random_split(
      dataset, [train_size, val_size, test_size]
  )

  train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
  val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
  test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

  return train_loader, val_loader, test_loader
