# Bhumika Yadav , Ishan Chaudhary
# Fall 2025
# CS 5330 Computer Vision

# examine_network.py
# Task 2: Analyze the trained MNIST CNN model

import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
import cv2
import numpy as np

from train_mnist import MyNetwork   # import your model definition


# ----------------------------
# Load trained model
# ----------------------------
model_path = "/Users/bhumikayadav/Desktop/Project 5 /mnist_cnn.pth"

model = MyNetwork()
model.load_state_dict(torch.load(model_path, map_location="cpu"))
model.eval()

print("\n===== MODEL STRUCTURE =====\n")
print(model)


# ----------------------------
# Load 1 sample from MNIST train set
# ----------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = torchvision.datasets.MNIST(
    "./data", train=True, download=True, transform=transform
)

first_image, first_label = train_dataset[0]
first_image_np = first_image.numpy().squeeze()  # shape (28, 28)


# ----------------------------
# Examine conv1 weights
# ----------------------------
with torch.no_grad():
    conv1_weights = model.conv1.weight.clone()  # shape [10, 1, 5, 5]

print("\n===== CONV1 WEIGHTS SHAPE =====")
print(conv1_weights.shape)   # should be [10, 1, 5, 5]

print("\n===== FIRST FILTER (weights[0, 0]) =====")
print(conv1_weights[0, 0])


# ----------------------------
# Visualize the 10 filters
# ----------------------------
plt.figure(figsize=(8, 6))
for i in range(10):
    plt.subplot(3, 4, i+1)
    plt.imshow(conv1_weights[i, 0], cmap="gray")
    plt.title(f"Filter {i}")
    plt.xticks([])
    plt.yticks([])

plt.tight_layout()
plt.savefig("conv1_filters.png")
plt.close()
print("Saved conv1 filter visualization → conv1_filters.png")


# ----------------------------
# Apply filters to the first image
# (using OpenCV filter2D)
# ----------------------------
first_img_for_cv = first_image_np  # already normalized but OK for visualization

plt.figure(figsize=(8, 6))
for i in range(10):
    kernel = conv1_weights[i, 0].numpy()

    # OpenCV expects float32
    kernel = kernel.astype(np.float32)

    # apply filter
    filtered = cv2.filter2D(first_img_for_cv, -1, kernel)

    plt.subplot(3, 4, i+1)
    plt.imshow(filtered, cmap="gray")
    plt.title(f"Filtered {i}")
    plt.xticks([])
    plt.yticks([])

plt.tight_layout()
plt.savefig("conv1_filterResults.png")
plt.close()
print("Saved filtered image visualization → conv1_filterResults.png")

print("\nTask 2 complete!")
