# Bhumika Yadav , Ishan Chaudhary
# Fall 2025
# CS 5330 Computer Vision

# test_mnist.py
# Loads trained MNIST CNN, evaluates first 10 test images, prints outputs as table, plots first 9 digits with predictions.

import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
import pandas as pd

# -----------------------
# Network definition 
# -----------------------
class MyNetwork(nn.Module):
    def __init__(self):
        super(MyNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d(p=0.25)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
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
def get_test_loader(batch_size_test=1000, data_dir="./data"):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(data_dir, train=False, download=True, transform=transform),
        batch_size=batch_size_test, shuffle=False
    )
    return test_loader

def test_first_10(network, device, test_loader):
    network.eval()
    data_iter = iter(test_loader)
    images, labels = next(data_iter)
    images, labels = images.to(device), labels.to(device)
    
    print("\nFirst 10 test images predictions:")
    outputs_list = []
    with torch.no_grad():
        output = network(images[:10])
        for i in range(10):
            out_vals = output[i].cpu().numpy()
            pred_label = int(out_vals.argmax())
            outputs_list.append((i+1, pred_label, labels[i].item(), out_vals))
    
    # Print as table
    data = []
    for img_idx, pred, true, out_vals in outputs_list:
        row = [img_idx, pred, true] + ["{:.2f}".format(x) for x in out_vals]
        data.append(row)
    cols = ["Image", "Pred", "True"] + [f"Class {j}" for j in range(10)]
    df = pd.DataFrame(data, columns=cols)
    print(df.to_string(index=False))
    
    return images[:9], output[:9], labels[:9]

def plot_first_9(images, output, labels, out_path="first_9_predictions.png"):
    fig = plt.figure(figsize=(6,6))
    for i in range(9):
        plt.subplot(3,3,i+1)
        plt.tight_layout()
        plt.imshow(images[i][0].cpu(), cmap='gray', interpolation='none')
        pred_label = int(output[i].cpu().numpy().argmax())
        true_label = labels[i].item()
        plt.title(f"Pred: {pred_label}\nTrue: {true_label}")
        plt.xticks([])
        plt.yticks([])
    plt.suptitle("First 9 MNIST test digits with predictions")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved first 9 predictions plot to {out_path}")

# -----------------------
# Main
# -----------------------
def main(argv):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Path to saved model
    model_path = "./mnist_cnn.pth"
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found.")
        return

    # Load test data
    test_loader = get_test_loader(batch_size_test=1000)

    # Load model
    network = MyNetwork().to(device)
    network.load_state_dict(torch.load(model_path, map_location=device))
    network.eval()
    print(f"Loaded model from {model_path}")

    # Test first 10 and plot first 9
    images9, output9, labels9 = test_first_10(network, device, test_loader)
    plot_first_9(images9, output9, labels9)

if __name__ == "__main__":
    main(sys.argv)

