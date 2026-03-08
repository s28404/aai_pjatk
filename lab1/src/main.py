import argparse
import os
import torch
from data import WineDataModule
from model import MLPV1, MLPV2
from training import MLPTrainer
from testing import Tester
from utils import save_stats, plot_loss, load_hyperparameters


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


def main(args):
    config = load_hyperparameters("hyperparameters.yaml")
    data_module = WineDataModule(batch_size=config["batch_size"])
    data_module.prepare_data()
    data_module.setup()

    if args.model_type == "mlp":
        input_dim = data_module.train_ds.dataset.tensors[0].shape[1]
        output_dim = len(set(data_module.train_ds.dataset.tensors[1].numpy()))
        if args.model_version == "v1":
            model = MLPV1(input_dim, output_dim)
        elif args.model_version == "v2":
            model = MLPV2(input_dim, output_dim)
        else:
            raise ValueError("Invalid model version. Choose 'v1' or 'v2'.")

        optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
        criterion = torch.nn.CrossEntropyLoss()

        trainer = MLPTrainer(
            model=model,
            data_module=data_module,
            optimizer=optimizer,
            criterion=criterion,
            device="cuda" if torch.cuda.is_available() else "cpu",
            num_epochs=args.num_epochs,
            version=args.model_version,
        )
        stats = trainer.fit()

        os.makedirs("plots", exist_ok=True)
        plot_loss(
            stats["avg_losses"],
            f"plots/loss_curve_{args.model_version}_{args.num_epochs}.png",
        )
        tester = Tester(
            model, data_module, device="cuda" if torch.cuda.is_available() else "cpu"
        )
        test_accuracy = tester.test()
        stats["accuracy"] = test_accuracy
        print(f"Test Accuracy for {args.model_version}: {test_accuracy:.4f}")

        os.makedirs("stats", exist_ok=True)
        save_stats(stats, f"stats/{args.model_version}_{args.num_epochs}.json")

    elif args.model_type == "knn":
        X_train, y_train = data_module.train_ds.dataset.tensors
        X_test, y_test = data_module.test_ds.dataset.tensors

        X_train, y_train = X_train.numpy(), y_train.numpy()
        X_test, y_test = X_test.numpy(), y_test.numpy()

        knn = KNeighborsClassifier(n_neighbors=args.n_neighbors)
        knn.fit(X_train, y_train)

        y_pred = knn.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_pred)
        print(f"Test Accuracy for KNN: {test_accuracy:.4f}")

        stats = {"accuracy": test_accuracy, "n_neighbors": args.n_neighbors}
        os.makedirs("stats", exist_ok=True)
        save_stats(stats, f"stats/knn_{args.n_neighbors}.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train model on Wine dataset")
    parser.add_argument(
        "--model_type",
        type=str,
        default="mlp",
        help="Model type to train (mlp or knn)",
    )
    parser.add_argument(
        "--model_version",
        type=str,
        default="v1",
        help="Model version to train (v1 or v2, only for mlp)",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=100,
        help="Number of training epochs (only for mlp)",
    )

    parser.add_argument(
        "--n_neighbors",
        type=int,
        default=3,
        help="Number of neighbors for KNN (only for knn)",
    )
    args = parser.parse_args()

    main(args)
