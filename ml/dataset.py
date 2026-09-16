import torch
from torch.utils.data import Dataset


class TabularDataset(Dataset):
    def __init__(
        self,
        numeric_data,
        categorical_data,
        target,
    ):
        self.numeric = torch.tensor(
            numeric_data,
            dtype=torch.float32,
        )

        self.categorical = torch.tensor(
            categorical_data,
            dtype=torch.long,
        )

        self.target = torch.tensor(
            target,
            dtype=torch.float32,
        )

    def __len__(self):
        return len(self.target)

    def __getitem__(self, index):
        return {
            "numeric": self.numeric[index],
            "categorical": self.categorical[index],
            "target": self.target[index],
        }