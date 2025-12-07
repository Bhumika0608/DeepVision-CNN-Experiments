# Bhumika Yadav , Ishan Chaudhary
# Fall 2025
# CS 5330 Computer Vision

# Fashion MNIST Deep Learning Project
# Extension : This builds, trains, evaluates, and analyzes a CNN for Fashion MNIST digit recognition.
# Added functionality to save plots as images.

import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os

# ----------------------
# Network Definition
# ----------------------
class MyNetwork(nn.Module):
    def __init__(self):
        super(MyNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.dropout = nn.Dropout(0.5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(20*4*4, 50)
        self.fc2 = nn.Linear(50, 10)
        self.log_softmax = nn.LogSoftmax(dim=1)
    
    def forward(self, x):
        x = torch.relu(self.pool1(self.conv1(x)))
        x = torch.relu(self.pool2(self.dropout(self.conv2(x))))
        x = x.view(-1, 20*4*4)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.log_softmax(x)
        return x

# ----------------------
# Training function
# ----------------------
def train_network(batch_size=64, epochs=5, lr=0.01):
    transform = transforms.ToTensor()
    
    train_set = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    test_set = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MyNetwork().to(device)
    criterion = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    
    train_acc_list = []
    test_acc_list = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        model.eval()
        correct_train, total_train = 0, 0
        correct_test, total_test = 0, 0
        
        with torch.no_grad():
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total_train += labels.size(0)
                correct_train += (predicted == labels).sum().item()
            
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total_test += labels.size(0)
                correct_test += (predicted == labels).sum().item()
        
        train_acc = 100 * correct_train / total_train
        test_acc = 100 * correct_test / total_test
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}, "
              f"Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%")
    
    # Save model
    torch.save(model.state_dict(), "fashion_mnist_model.pth")
    print("Model saved as fashion_mnist_model.pth")
    
    # Plot training & test accuracy and save image
    plt.figure(figsize=(8,5))
    plt.plot(range(1, epochs+1), train_acc_list, label='Train Accuracy', marker='o')
    plt.plot(range(1, epochs+1), test_acc_list, label='Test Accuracy', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Test Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig("accuracy_plot.png")  # Save as image
    plt.close()
    print("Accuracy plot saved as accuracy_plot.png")
    
    return model

# ----------------------
# Visualize first layer filters and save
# ----------------------
def visualize_first_layer(model):
    with torch.no_grad():
        filters = model.conv1.weight.cpu().numpy()
        num_filters = filters.shape[0]
        fig, axes = plt.subplots(1, num_filters, figsize=(15, 2))
        for i in range(num_filters):
            axes[i].imshow(filters[i, 0], cmap='gray')
            axes[i].axis('off')
            axes[i].set_title(f"F{i}")
        plt.suptitle("First Conv Layer Filters")
        plt.savefig("conv1_filters.png")  # Save as image
        plt.close()
        print("First conv layer filters saved as conv1_filters.png")

# ----------------------
# Main function
# ----------------------
def main(argv):
    # Create output folder if needed
    os.makedirs("results", exist_ok=True)
    model = train_network(batch_size=64, epochs=5, lr=0.01)
    visualize_first_layer(model)

if __name__ == "__main__":
    main(sys.argv)
