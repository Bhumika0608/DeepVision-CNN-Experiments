# Bhumika Yadav , Ishan Chaudhary
# Fall 2025
# CS 5330 Computer Vision

# Test Transfer Learning Model on Custom Greek Letter Images
# Loads trained Greek letter model and tests on user-provided images

import sys
import os
import argparse
from typing import Tuple, List
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageOps
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from train_mnist import MyNetwork


# -----------------------
# Network Definitions
# -----------------------
class GreekTransform:
    """
    Transform Greek letter images to MNIST format:
    - Invert RGB first (normalize to white on black)
    - Convert RGB to grayscale
    - Scale and center-crop to 28x28
    """
    def __init__(self):
        pass

    def __call__(self, x):
        x = torchvision.transforms.functional.invert(x)  # Invert RGB first
        x = torchvision.transforms.functional.rgb_to_grayscale(x)
        x = torchvision.transforms.functional.affine(x, 0, (0, 0), 36/128, 0)
        x = torchvision.transforms.functional.center_crop(x, (28, 28))
        return x


class GreekNetwork(nn.Module):
    """
    Transfer learning network for Greek letters (3 outputs).
    """
    def __init__(self, pretrained_model: MyNetwork):
        super(GreekNetwork, self).__init__()
        self.conv1 = pretrained_model.conv1
        self.conv2 = pretrained_model.conv2
        self.conv2_drop = pretrained_model.conv2_drop
        self.fc1 = pretrained_model.fc1
        self.fc2 = nn.Linear(50, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
def load_greek_model(model_path: str, mnist_model_path: str, device) -> GreekNetwork:
    """
    Load the trained Greek letter model.
    """
    # First load the pre-trained MNIST model structure
    pretrained = MyNetwork().to(device)
    if os.path.exists(mnist_model_path):
        pretrained.load_state_dict(torch.load(mnist_model_path, map_location=device))
    
    # Create Greek network and load weights
    model = GreekNetwork(pretrained).to(device)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded Greek letter model from {model_path}")
    else:
        print(f"Error: Model file {model_path} not found.")
        return None
    
    model.eval()
    return model


def preprocess_image(image_path: str) -> Tuple[torch.Tensor, Image.Image]:
    """
    Load and preprocess a Greek letter image.
    Returns both the preprocessed tensor and original image.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Load original image for display
    original_img = Image.open(image_path).convert("RGB")
    
    # Preprocess for network
    img = Image.open(image_path).convert("RGB")
    
    # Apply transformations
    tensor = transforms.ToTensor()(img)
    tensor = GreekTransform()(tensor)
    tensor = transforms.Normalize((0.1307,), (0.3081,))(tensor)
    tensor = tensor.unsqueeze(0)  # Add batch dimension
    
    return tensor, original_img


def predict_greek_letter(model: GreekNetwork, device, image_tensor: torch.Tensor) -> Tuple[str, float, List[float]]:
    """
    Predict Greek letter from image tensor.
    Returns: (letter_name, confidence, all_outputs)
    """
    greek_letters = ["alpha", "beta", "gamma"]
    
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        output = model(image_tensor)
        probs = torch.exp(output[0])  # Convert log_softmax to probabilities
        confidence, pred_idx = torch.max(probs, 0)
        pred_letter = greek_letters[pred_idx.item()]
        outputs = probs.cpu().numpy()
    
    return pred_letter, confidence.item(), outputs


def test_custom_images(model: GreekNetwork, device, image_paths: List[str], 
                      plot_path: str = "greek_predictions.png") -> None:
    """
    Test model on custom Greek letter images and display results.
    """
    greek_letters = ["alpha", "beta", "gamma"]
    num_images = len(image_paths)
    
    predictions = []
    images = []
    
    print(f"\nTesting on {num_images} custom Greek letter images:")
    print("-" * 60)
    
    for i, img_path in enumerate(image_paths, 1):
        try:
            tensor, original_img = preprocess_image(img_path)
            pred_letter, confidence, outputs = predict_greek_letter(model, device, tensor)
            
            predictions.append({
                'path': img_path,
                'prediction': pred_letter,
                'confidence': confidence,
                'outputs': outputs
            })
            images.append(original_img)
            
            print(f"\nImage {i}: {os.path.basename(img_path)}")
            print(f"  Prediction: {pred_letter.upper()}")
            print(f"  Confidence: {confidence:.2%}")
            print(f"  Outputs: alpha={outputs[0]:.4f}, beta={outputs[1]:.4f}, gamma={outputs[2]:.4f}")
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    
    # Plot results
    if len(images) > 0:
        num_cols = min(3, len(images))
        num_rows = (len(images) + num_cols - 1) // num_cols
        
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 4*num_rows))
        if num_rows == 1 and num_cols == 1:
            axes = [axes]
        elif num_rows == 1 or num_cols == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
        
        for idx, (pred, ax) in enumerate(zip(predictions, axes)):
            ax.imshow(images[idx])
            ax.set_title(f"Pred: {pred['prediction'].upper()}\n(Conf: {pred['confidence']:.1%})", fontsize=12)
            ax.axis('off')
        
        # Hide any unused subplots
        for ax in axes[len(predictions):]:
            ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(plot_path, bbox_inches='tight', dpi=100)
        plt.close()
        print(f"\nSaved predictions plot to {plot_path}")


# -----------------------
# Main Function
# -----------------------
def main(argv):
    """
    Main function to test Greek letter model on custom images.
    """
    parser = argparse.ArgumentParser(description="Test Greek letter transfer learning model")
    parser.add_argument('--greek-model', type=str, default='./greek_cnn.pth', help='Path to trained Greek model')
    parser.add_argument('--mnist-model', type=str, default='./mnist_cnn.pth', help='Path to pre-trained MNIST model')
    parser.add_argument('--images', type=str, nargs='+', help='Paths to Greek letter images to test')
    parser.add_argument('--image-dir', type=str, help='Directory containing Greek letter images')
    parser.add_argument('--plot-output', type=str, default='greek_predictions.png', help='Output plot filename')
    
    args = parser.parse_args(argv[1:])
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    model = load_greek_model(args.greek_model, args.mnist_model, device)
    if model is None:
        return
    
    # Collect image paths
    image_paths = []
    
    if args.images:
        image_paths.extend(args.images)
    
    if args.image_dir:
        if os.path.isdir(args.image_dir):
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
                image_paths.extend(sorted(Path(args.image_dir).glob(ext)))
        else:
            print(f"Error: Directory {args.image_dir} not found")
            return
    
    if not image_paths:
        print("Error: No images provided. Use --images or --image-dir")
        return
    
    # Test on images
    test_custom_images(model, device, image_paths, args.plot_output)


if __name__ == "__main__":
    main(sys.argv)
