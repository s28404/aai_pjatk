import torch
import os
from tqdm import tqdm
from sklearn.metrics import accuracy_score


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

    def evaluate_accuracy(self, dataloader):
        self.model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for X_batch, y_batch in dataloader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                outputs = self.model(
                    X_batch
                )  # outputs.shape = (batch_size, num_classes)
                preds = torch.argmax(
                    outputs, dim=1
                )  # take the index of the max log-probability as the predicted class
                all_preds.extend(
                    preds.cpu().numpy()
                )  # move to CPU and convert to numpy array for accuracy_score
                all_labels.extend(y_batch.cpu().numpy())
        return accuracy_score(all_labels, all_preds)

    def fit(self):
        stats = {"avg_losses": [], "accuracies": []}
        for epoch in tqdm(range(self.num_epochs), desc="Training"):
            avg_loss = self.train_epoch()
            train_accuracy = self.evaluate_accuracy(self.data_module.train_dataloader())
            stats["avg_losses"].append(avg_loss)
            stats["accuracies"].append(train_accuracy)
            print(
                f"Epoch {epoch + 1}/{self.num_epochs}, Loss: {avg_loss:.4f}, Accuracy: {train_accuracy:.4f}"
            )

        os.makedirs("models", exist_ok=True)
        torch.save(self.model.state_dict(), f"models/mlp_model_{self.version}.pth")

        return stats
