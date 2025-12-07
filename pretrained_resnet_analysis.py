# Bhumika Yadav , Ishan Chaudhary
# Fall 2025
# CS 5330 Computer Vision

# Extension: Pre-trained ResNet18 - First Convolution Layer Analysis
# This script loads a pre-trained ResNet18, visualizes its first conv layer filters,
# and applies the filters to a sample image, saving the outputs as images.

import torch
import torchvision.models as models
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import cv2
from torchvision.datasets import CIFAR10
from PIL import Image
import os

# ----------------------
# Load pre-trained network
# ----------------------
def load_pretrained_resnet():
    model = models.resnet18(pretrained=True)
    model.eval()  # set to evaluation mode
    print("ResNet18 structure:\n")
    print(model)
    return model

# ----------------------
# Visualize first conv layer filters
# ----------------------
def visualize_conv1_filters(model, save_path="resnet18_conv1_filters.png"):
    conv1_weights = model.conv1.weight.detach().cpu().numpy()  # [64,3,7,7]
    num_filters = 16  # visualize first 16 filters

    fig, axes = plt.subplots(4,4, figsize=(8,8))
    for i in range(num_filters):
        # convert 3-channel filter to grayscale for visualization
        filt = conv1_weights[i].mean(axis=0)
        ax = axes[i//4, i%4]
        ax.imshow(filt, cmap='gray')
        ax.axis('off')
        ax.set_title(f'F{i}')
    plt.suptitle("First Conv Layer Filters - ResNet18")
    plt.savefig(save_path)
    plt.close()
    print(f"First conv layer filters saved as {save_path}")

# ----------------------
# Apply filters to a sample image
# ----------------------
def apply_filters_to_image(model, save_path="resnet18_conv1_filterResults.png"):
    # Load a sample image from CIFAR10
    cifar_test = CIFAR10(root='./data', train=False, download=True)
    img, label = cifar_test[0]  # get first image
    img_gray = np.array(img.convert('L'))  # convert to grayscale

    conv1_weights = model.conv1.weight.detach().cpu().numpy()

    fig, axes = plt.subplots(2,5, figsize=(12,5))
    for i in range(10):  # apply first 10 filters
        filt = conv1_weights[i].mean(axis=0)  # convert 3-channel filter to single channel
        filtered_img = cv2.filter2D(img_gray, -1, filt)
        ax = axes[i//5, i%5]
        ax.imshow(filtered_img, cmap='gray')
        ax.axis('off')
        ax.set_title(f'F{i}')
    plt.suptitle("Effect of ResNet18 conv1 filters on sample image")
    plt.savefig(save_path)
    plt.close()
    print(f"Filtered images saved as {save_path}")

# ----------------------
# Main function
# ----------------------
def main():
    os.makedirs("results", exist_ok=True)
    model = load_pretrained_resnet()
    visualize_conv1_filters(model, save_path="results/resnet18_conv1_filters.png")
    apply_filters_to_image(model, save_path="results/resnet18_conv1_filterResults.png")

if __name__ == "__main__":
    main()
