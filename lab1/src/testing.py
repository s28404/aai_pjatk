import torch
from sklearn.metrics import accuracy_score


class Tester:
    def __init__(self, model, data_module, device="cuda"):
        self.model = model.to(device)
        self.data_module = data_module
        self.device = device

    def test(self):
        self.model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for X_batch, y_batch in self.data_module.test_dataloader():
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                outputs = self.model(X_batch)
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y_batch.cpu().numpy())
        return accuracy_score(all_labels, all_preds)
