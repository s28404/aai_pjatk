import json
import matplotlib

matplotlib.use("Agg")  # Use a non-interactive backend for plotting
import matplotlib.pyplot as plt
import yaml


def load_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def save_stats(stats, filepath):
    with open(filepath, "w") as f:
        json.dump(stats, f)


def plot_loss(X, filepath):
    plt.figure(figsize=(10, 5))
    plt.plot(X, label="Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Over Epochs")
    plt.legend()
    plt.grid()
    plt.savefig(filepath)
    plt.close()


def plot_accuracy(X, filepath):
    plt.figure(figsize=(10, 5))
    plt.plot(X, label="Train Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training Accuracy Over Epochs")
    plt.legend()
    plt.grid()
    plt.savefig(filepath)
    plt.close()
