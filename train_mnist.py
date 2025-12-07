# Bhumika Yadav , Ishan Chaudhary
# Fall 2025
# CS 5330 Computer Vision

# MNIST CNN Project
# Trains a small CNN on MNIST, saves model and produces training/test loss & accuracy plots.
# Usage:
# python3 train_mnist.py --epochs 15 --batch-size-train 64 --save-path ./mnist_cnn.pth

import sys
import argparse
import os
from typing import Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt


# -----------------------
# Network definition
# -----------------------
class MyNetwork(nn.Module):
    """
    Convolutional network:
      - Conv2d 1 -> 10, kernel 5
      - ReLU + MaxPool2d(2)
      - Conv2d 10 -> 20, kernel 5
      - Dropout2d(0.25)
      - ReLU + MaxPool2d(2)
      - Flatten
      - Linear(320 -> 50) + ReLU
      - Dropout(0.25)
      - Linear(50 -> 10)
      - LogSoftmax (dim=1)
    """

    def __init__(self):
        super(MyNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d(p=0.25)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute forward pass through the network.
        """
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training, p=0.25)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


# -----------------------
# Utility functions
# -----------------------
def get_data_loaders(batch_size_train: int, batch_size_test: int, data_dir: str = "./data") -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Prepare train and test data loaders for MNIST.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(data_dir, train=True, download=True, transform=transform),
        batch_size=batch_size_train, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(data_dir, train=False, download=True, transform=transform),
        batch_size=batch_size_test, shuffle=False  # shuffle - false
    )
    return train_loader, test_loader


def show_first_six_test_images(test_loader: torch.utils.data.DataLoader, out_path: str = "first_six_test.png") -> None:
    """
    Plot the first six images from the test set (not shuffled) and save the figure.
    """
    examples = enumerate(test_loader)
    batch_idx, (example_data, example_targets) = next(examples)
    fig = plt.figure(figsize=(6, 4))
    for i in range(6):
        plt.subplot(2, 3, i + 1)
        plt.tight_layout()
        plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
        plt.title(f"GT: {example_targets[i].item()}")
        plt.xticks([])
        plt.yticks([])
    plt.suptitle("First 6 examples from MNIST test set")
    plt.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved first-six-test image to {out_path}")


def train_one_epoch(network: nn.Module, device: torch.device, train_loader: torch.utils.data.DataLoader,
                    optimizer: optim.Optimizer, epoch: int, log_interval: int = 100) -> Tuple[float, float, List[float]]:
    """
    Train the network for one epoch. Returns (average_loss, accuracy_percent, per-batch-loss-list).
    """
    network.train()
    correct = 0
    train_loss_sum = 0.0
    batch_losses = []
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = network(data)
        loss = F.nll_loss(output, target, reduction='sum')
        loss.backward()
        optimizer.step()

        train_loss_sum += loss.item()
        pred = output.argmax(dim=1, keepdim=False)
        correct += pred.eq(target).sum().item()

        if batch_idx % log_interval == 0:
            batch_loss = loss.item() / len(data)
            batch_losses.append(batch_loss)
            print(f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} "
                  f"({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {batch_loss:.6f}")

    avg_loss = train_loss_sum / len(train_loader.dataset)
    accuracy = 100.0 * correct / len(train_loader.dataset)
    return avg_loss, accuracy, batch_losses


def evaluate(network: nn.Module, device: torch.device, test_loader: torch.utils.data.DataLoader) -> Tuple[float, float]:
    """
    Evaluate network on the test set. Returns (average_loss, accuracy_percent).
    """
    network.eval()
    test_loss_sum = 0.0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = network(data)
            test_loss_sum += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=False)
            correct += pred.eq(target).sum().item()
    avg_loss = test_loss_sum / len(test_loader.dataset)
    accuracy = 100.0 * correct / len(test_loader.dataset)
    return avg_loss, accuracy


def plot_metrics(train_losses: List[float], test_losses: List[float],
                 train_acc: List[float], test_acc: List[float],
                 out_prefix: str = "metrics"):
    """
    Plot and save training/test loss and accuracy.
    """
    # Loss plot
    fig1 = plt.figure()
    plt.plot(range(1, len(train_losses) + 1), train_losses, label="Train Loss")
    plt.plot(range(1, len(test_losses) + 1), test_losses, label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Average Loss")
    plt.legend()
    plt.title("Loss per epoch")
    plt.savefig(f"{out_prefix}_loss.png")
    plt.close(fig1)

    # Accuracy plot
    fig2 = plt.figure()
    plt.plot(range(1, len(train_acc) + 1), train_acc, label="Train Accuracy")
    plt.plot(range(1, len(test_acc) + 1), test_acc, label="Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.title("Accuracy per epoch")
    plt.savefig(f"{out_prefix}_accuracy.png")
    plt.close(fig2)
    print(f"Saved metrics plots: {out_prefix}_loss.png, {out_prefix}_accuracy.png")


# -----------------------
# Main
# -----------------------
def main(argv):
    """
    Main function: parses args, trains the model for N epochs, saves model and plots.
    """
    parser = argparse.ArgumentParser(description="Train MNIST CNN and save model/plots.")
    parser.add_argument("--epochs", type=int, default=15, help="Number of epochs (>=5 recommended).")
    parser.add_argument("--batch-size-train", type=int, default=64, help="Training batch size.")
    parser.add_argument("--batch-size-test", type=int, default=1000, help="Test batch size.")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate.")
    parser.add_argument("--momentum", type=float, default=0.0, help="SGD momentum (ignored for Adam).")
    parser.add_argument("--log-interval", type=int, default=100, help="How many batches between log prints.")
    parser.add_argument("--save-path", type=str, default="./mnist_cnn.pth", help="Where to save the trained model.")
    parser.add_argument("--data-dir", type=str, default="./data", help="Where to put MNIST data.")
    args = parser.parse_args(argv[1:])

    # Deterministic-ish settings
    random_seed = 1
    torch.manual_seed(random_seed)
    torch.backends.cudnn.enabled = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, test_loader = get_data_loaders(args.batch_size_train, args.batch_size_test, data_dir=args.data_dir)

    # show first six test images (non-shuffled)
    show_first_six_test_images(test_loader, out_path="first_six_test.png")

    network = MyNetwork().to(device)
    optimizer = optim.Adam(network.parameters(), lr=args.lr)

    train_losses = []
    test_losses = []
    train_acc = []
    test_acc = []

    # initial evaluation before training
    init_test_loss, init_test_acc = evaluate(network, device, test_loader)
    print(f"Initial Test: loss={init_test_loss:.4f}, acc={init_test_acc:.2f}%")
    test_losses.append(init_test_loss)
    test_acc.append(init_test_acc)

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc, _ = train_one_epoch(network, device, train_loader, optimizer, epoch, log_interval=args.log_interval)
        te_loss, te_acc = evaluate(network, device, test_loader)

        train_losses.append(tr_loss)
        test_losses.append(te_loss)
        train_acc.append(tr_acc)
        test_acc.append(te_acc)

        print(f"Epoch {epoch} summary: Train loss={tr_loss:.4f}, Train acc={tr_acc:.2f}%; Test loss={te_loss:.4f}, Test acc={te_acc:.2f}%")

        # save checkpoint after each epoch
        os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
        torch.save(network.state_dict(), args.save_path)
        print(f"Saved model state to {args.save_path}")

    plot_metrics(train_losses, test_losses, train_acc, test_acc, out_prefix="training_metrics")
    print("Training complete.")


if __name__ == "__main__":
    main(sys.argv)
