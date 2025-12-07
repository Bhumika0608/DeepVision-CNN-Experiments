# Bhumika Yadav , Ishan Chaudhary
# Fall 2025
# CS 5330 Computer Vision

# test_mnist_new_inputs.py
# Test trained MNIST CNN on individual handwritten digit images (0–9)

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageOps
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# ----------------------------------------------------
# Network (same architecture used during training)
# ----------------------------------------------------
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
        x = F.dropout(x, p=0.25, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# ----------------------------------------------------
# Image Preprocessing
# ----------------------------------------------------
def preprocess_image(path):
    img = Image.open(path).convert("L")   # convert to grayscale
    img = ImageOps.invert(img)            # invert to match MNIST
    img = img.resize((28, 28))            # resize to 28x28
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    tensor = transform(img).unsqueeze(0)  # add batch dimension
    return tensor, img

# ----------------------------------------------------
# Prediction
# ----------------------------------------------------
def predict(network, device, tensor):
    network.eval()
    with torch.no_grad():
        output = network(tensor.to(device))
        pred = output.argmax(dim=1).item()
    return pred

# ----------------------------------------------------
# Main
# ----------------------------------------------------
def main():
    IMAGES_FOLDER = "./my_digits"  # folder with images
    MODEL_PATH = "./mnist_cnn.pth"
    OUTPUT_FOLDER = "./predicted_digits"
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network = MyNetwork().to(device)
    network.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    print(f"Loaded trained model from: {MODEL_PATH}\n")

    # Process each image individually
    for img_file in sorted(os.listdir(IMAGES_FOLDER)):
        img_path = os.path.join(IMAGES_FOLDER, img_file)
        if not img_file.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        tensor, img_display = preprocess_image(img_path)
        pred = predict(network, device, tensor)

        print(f"{img_file} → Predicted: {pred}")

        # Save prediction plot
        plt.imshow(img_display, cmap="gray")
        plt.title(f"Predicted: {pred}")
        plt.axis("off")
        save_path = os.path.join(OUTPUT_FOLDER, f"pred_{img_file}")
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print(f"Saved prediction plot to {save_path}")

if __name__ == "__main__":
    main()
