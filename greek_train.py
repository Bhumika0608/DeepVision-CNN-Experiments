# Bhumika Yadav , Ishan Chaudhary
# Fall 2025
# CS 5330 Computer Vision

# Transfer Learning on Greek Letters
# Reuses the pre-trained MNIST CNN to recognize Greek letters (alpha, beta, gamma)
# Trains on 27 examples (9 of each letter) and evaluates transfer learning performance

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

from train_mnist import MyNetwork


# -----------------------
# Greek Transform Class
# -----------------------
class GreekTransform:
    """
    Transform Greek letter images (133x133 RGB) to match MNIST format:
    - Convert RGB to grayscale
    - Scale and center-crop to 28x28
    - Invert intensities (Greek dataset: black on white -> white on black)
    """
    def __init__(self):
        pass

    def __call__(self, x):
        """
        Apply transformations to Greek letter image.
        """
        x = torchvision.transforms.functional.rgb_to_grayscale(x)
        x = torchvision.transforms.functional.affine(x, 0, (0, 0), 36/128, 0)
        x = torchvision.transforms.functional.center_crop(x, (28, 28))
        return torchvision.transforms.functional.invert(x)


# -----------------------
# Modified Network for Greek Letters
# -----------------------
class GreekNetwork(nn.Module):
    """
    Transfer learning network: uses MNIST pre-trained feature extractor
    with a new 3-output classification head for Greek letters (alpha=0, beta=1, gamma=2).
    """
    def __init__(self, pretrained_model: MyNetwork):
        super(GreekNetwork, self).__init__()
        # Copy all layers from pre-trained MNIST model except the last layer
        self.conv1 = pretrained_model.conv1
        self.conv2 = pretrained_model.conv2
        self.conv2_drop = pretrained_model.conv2_drop
        self.fc1 = pretrained_model.fc1
        
        # Replace the last layer with 3 outputs for Greek letters
        self.fc2 = nn.Linear(50, 3)

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
# Utility Functions
# -----------------------
def load_pretrained_model(model_path: str, device) -> MyNetwork:
    """
    Load the pre-trained MNIST model from file.
    """
    model = MyNetwork().to(device)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded pre-trained model from {model_path}")
    else:
        print(f"Warning: Model file {model_path} not found. Using randomly initialized model.")
    return model


def freeze_model_weights(model: nn.Module) -> None:
    """
    Freeze all parameters in the model (set requires_grad to False).
    This prevents the pre-trained weights from being updated during training.
    """
    for param in model.parameters():
        param.requires_grad = False
    print("Frozen all model parameters from pre-trained MNIST network")


def unfreeze_last_layer(model: GreekNetwork) -> None:
    """
    Unfreeze the last layer (fc2) so it can be trained on Greek letters.
    """
    for param in model.fc2.parameters():
        param.requires_grad = True
    print("Unfroze the last layer (fc2) for Greek letter training")


def get_greek_data_loader(training_set_path: str, batch_size: int = 5, shuffle: bool = True):
    """
    Create DataLoader for Greek letter dataset using ImageFolder.
    Applies GreekTransform and normalizes to MNIST statistics.
    """
    greek_loader = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(
            training_set_path,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                GreekTransform(),
                torchvision.transforms.Normalize((0.1307,), (0.3081,))
            ])
        ),
        batch_size=batch_size,
        shuffle=shuffle
    )
    return greek_loader


def train_epoch(model: nn.Module, device, train_loader, optimizer, epoch: int) -> float:
    """
    Train the model for one epoch on the Greek letter dataset.
    Returns the average loss for the epoch.
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    avg_loss = total_loss / num_batches
    print(f"Epoch {epoch}: Average Loss = {avg_loss:.4f}")
    return avg_loss


def evaluate(model: nn.Module, device, test_loader) -> Tuple[float, float]:
    """
    Evaluate the model on the test set.
    Returns accuracy and average loss.
    """
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = F.nll_loss(output, target)
            total_loss += loss.item()
            
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)
    
    accuracy = 100.0 * correct / total if total > 0 else 0.0
    avg_loss = total_loss / len(test_loader) if len(test_loader) > 0 else 0.0
    
    return accuracy, avg_loss


def plot_training_loss(losses: List[float], out_path: str = "greek_training_loss.png") -> None:
    """
    Plot training loss across epochs and save to file.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(losses) + 1), losses, marker='o', linestyle='-', linewidth=2, markersize=6)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.title("Greek Letter Transfer Learning - Training Loss", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight', dpi=100)
    plt.close()
    print(f"Saved training loss plot to {out_path}")


def save_model(model: GreekNetwork, out_path: str) -> None:
    """
    Save the trained Greek letter model to file.
    """
    torch.save(model.state_dict(), out_path)
    print(f"Saved trained model to {out_path}")


# -----------------------
# Main Function
# -----------------------
def main(argv):
    """
    Main function to run transfer learning on Greek letters.
    """
    parser = argparse.ArgumentParser(description="Transfer learning on Greek letters")
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=5, help='Batch size for training')
    parser.add_argument('--mnist-model', type=str, default='./mnist_cnn.pth', help='Path to pre-trained MNIST model')
    parser.add_argument('--greek-data', type=str, default='./greek_train', help='Path to Greek letter dataset')
    parser.add_argument('--save-model', type=str, default='./greek_cnn.pth', help='Path to save trained model')
    
    args = parser.parse_args(argv[1:])
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    # Load pre-trained MNIST model
    print("Loading pre-trained MNIST model...")
    pretrained_model = load_pretrained_model(args.mnist_model, device)
    print(pretrained_model)
    
    # Create transfer learning model
    print("\n" + "="*60)
    print("Creating transfer learning model for Greek letters...")
    print("="*60 + "\n")
    model = GreekNetwork(pretrained_model).to(device)
    
    # Freeze all weights except the last layer
    freeze_model_weights(model)
    unfreeze_last_layer(model)
    
    # Print modified model structure
    print("\nModified Network Structure:")
    print(model)
    
    # Load Greek letter dataset
    print(f"\nLoading Greek letter dataset from {args.greek_data}...")
    greek_loader = get_greek_data_loader(args.greek_data, batch_size=args.batch_size, shuffle=True)
    print(f"Loaded {len(greek_loader.dataset)} Greek letter images")
    
    # Create optimizer (only for fc2 layer since others are frozen)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    
    # Training loop
    print(f"\n" + "="*60)
    print(f"Training for {args.epochs} epochs...")
    print("="*60 + "\n")
    
    training_losses = []
    
    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, device, greek_loader, optimizer, epoch)
        training_losses.append(loss)
    
    # Evaluate on the training set (since we only have training data)
    accuracy, avg_loss = evaluate(model, device, greek_loader)
    print(f"\nFinal Accuracy on Greek dataset: {accuracy:.2f}%")
    print(f"Final Loss: {avg_loss:.4f}")
    
    # Save trained model
    save_model(model, args.save_model)
    
    # Generate training loss plot
    print("\nGenerating plots...")
    plot_training_loss(training_losses)
    
    print("\n" + "="*60)
    print("Greek letter transfer learning complete!")
    print("="*60)
    
    return


if __name__ == "__main__":
    main(sys.argv)
