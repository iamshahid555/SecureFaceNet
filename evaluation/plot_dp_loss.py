import matplotlib.pyplot as plt

def plot_dp_training_loss(losses):
    """
    Plot training loss over epochs for DP-SGD training.
    """
    epochs = range(1, len(losses) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, losses, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("DP-SGD Training Loss over Epochs")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Example losses from a DP training run
    dp_losses = [1.1071, 1.1041, 1.1002, 1.1002, 1.1005]

    plot_dp_training_loss(dp_losses)