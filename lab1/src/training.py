import torch
import os
from tqdm import tqdm


class MLPTrainer:
    def __init__(
        self,
        model,
        data_module,
        optimizer,
        criterion,
        device="cuda",
        num_epochs=100,
        version="v1",
    ):
        self.model = model.to(device)
        self.data_module = data_module
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.num_epochs = num_epochs
        self.version = version

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        for X_batch, y_batch in self.data_module.train_dataloader():
            X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(X_batch)
            loss = self.criterion(outputs, y_batch)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

        return total_loss / len(self.data_module.train_dataloader())

    def fit(self):
        stats = {
            "avg_losses": [],
            "version": self.version,
            "num_epochs": self.num_epochs,
            "accuracy": None,
        }
        for epoch in tqdm(range(self.num_epochs), desc="Training"):
            avg_loss = self.train_epoch()
            stats["avg_losses"].append(avg_loss)
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch {epoch + 1}/{self.num_epochs}, Loss: {avg_loss:.4f}")

        os.makedirs("models", exist_ok=True)
        torch.save(
            self.model.state_dict(),
            f"models/mlp_model_{self.version}_{self.num_epochs}.pth",
        )

        return stats
