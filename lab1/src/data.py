from sklearn.datasets import load_wine
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split
import pytorch_lightning as pl


class WineDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=32):
        super().__init__()
        self.batch_size = batch_size

    def prepare_data(self):
        load_wine()

    def setup(self):
        data = load_wine()
        X = torch.tensor(data.data, dtype=torch.float32)
        y = torch.tensor(data.target, dtype=torch.long)

        dataset = TensorDataset(X, y)
        train_size = int(0.8 * len(dataset))
        test_size = (
            len(dataset) - train_size
        )  # Ensure the sum matches the dataset length

        self.train_ds, self.test_ds = random_split(dataset, [train_size, test_size])

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=False)
