import argparse
import os
import torch
from data import WineDataModule
from model import MLPV1, MLPV2
from training import MLPTrainer
from utils import save_stats, plot_accuracy, plot_loss, load_config


def main(args):
    config = load_config(args.config_path)
    data_module = WineDataModule(batch_size=config["batch_size"])
    data_module.prepare_data()
    data_module.setup()

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

    os.makedirs("stats", exist_ok=True)
    save_stats(stats, f"stats/{args.model_version}.json")
    plot_loss(stats["avg_losses"]["train"], f"loss_curve_{args.model_version}.png")
    plot_accuracy(
        stats["avg_accuracies"]["train"], f"accuracy_curve_{args.model_version}.png"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MLP on Wine dataset")
    parser.add_argument(
        "--config_path", type=str, default="config.yaml", help="Path to config file"
    )
    parser.add_argument(
        "--model_version",
        type=str,
        default="v1",
        help="Model version to train (v1 or v2)",
    )
    parser.add_argument(
        "--num_epochs", type=int, default=100, help="Number of training epochs"
    )
    args = parser.parse_args()

    main(args)
