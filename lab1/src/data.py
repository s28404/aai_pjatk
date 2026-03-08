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

        self.train_ds = self.val_ds = random_split(dataset, [int(0.8 * len(dataset)), int(0.2 * len(dataset))])
    
    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True)