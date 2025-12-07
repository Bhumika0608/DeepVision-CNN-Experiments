# Bhumika Yadav , Ishan Chaudhary
# Fall 2025
# CS 5330 Computer Vision

# MNIST CNN with Gabor Filter Bank as First Layer

import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

# ----------------------
# Gabor filter bank generator
# ----------------------
def generate_gabor_filters(num_filters=10, kernel_size=5):
    filters = []
    for theta in np.linspace(0, np.pi, num_filters, endpoint=False):
        kern = cv2.getGaborKernel((kernel_size, kernel_size), sigma=1.0, theta=theta,
                                  lambd=3.0, gamma=0.5, psi=0, ktype=cv2.CV_32F)
        filters.append(kern)
    filters = np.stack(filters, axis=0)  # shape: [num_filters, k, k]
    filters = filters[:, np.newaxis, :, :]  # add channel dim: [num_filters,1,k,k]
    filters_tensor = torch.tensor(filters, dtype=torch.float32)
    return filters_tensor

# ----------------------
# Network definition
# ----------------------
class MyGaborNetwork(nn.Module):
    def __init__(self, gabor_filters):
        super(MyGaborNetwork, self).__init__()
        num_filters, in_channels, k, _ = gabor_filters.shape
        # First conv layer uses fixed Gabor filters
        self.conv1 = nn.Conv2d(in_channels, num_filters, kernel_size=k, bias=False)
        self.conv1.weight = nn.Parameter(gabor_filters, requires_grad=False)  # frozen
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(num_filters, 20, kernel_size=5)
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
    train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    gabor_filters = generate_gabor_filters(num_filters=10, kernel_size=5)
    model = MyGaborNetwork(gabor_filters).to(device)
    
    criterion = nn.NLLLoss()
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, momentum=0.9)
    
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
        
        # Evaluate
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
    torch.save(model.state_dict(), "mnist_gabor_model.pth")
    print("Model saved as mnist_gabor_model.pth")
    
    # Plot training/test accuracy
    plt.figure(figsize=(8,5))
    plt.plot(range(1, epochs+1), train_acc_list, label='Train Accuracy', marker='o')
    plt.plot(range(1, epochs+1), test_acc_list, label='Test Accuracy', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Test Accuracy with Gabor First Layer')
    plt.legend()
    plt.grid(True)
    plt.savefig("mnist_gabor_accuracy.png")
    plt.close()
    print("Accuracy plot saved as mnist_gabor_accuracy.png")
    
    return model

# ----------------------
# Main function
# ----------------------
def main(argv):
    os.makedirs("results", exist_ok=True)
    model = train_network(batch_size=64, epochs=5, lr=0.01)

if __name__ == "__main__":
    main(sys.argv)
