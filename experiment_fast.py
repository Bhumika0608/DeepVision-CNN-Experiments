# Bhumika Yadav , Ishan Chaudhary
# Fall 2025
# CS 5330 Computer Vision

# Task 4: Accelerated Network Architecture Experimentation
# Optimized for CPU execution with mock data for fast results


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import time
import csv
import os
from datetime import datetime
import argparse
import sys
import random
import numpy as np


class DynamicCNN(nn.Module):
    """Flexible CNN architecture"""
    
    def __init__(self, num_conv_layers: int = 2, base_filters: int = 32, dropout_rate: float = 0.25):
        super(DynamicCNN, self).__init__()
        
        self.num_conv_layers = num_conv_layers
        self.base_filters = base_filters
        self.dropout_rate = dropout_rate
        
        self.conv_blocks = nn.ModuleList()
        in_channels = 1
        
        for layer_idx in range(num_conv_layers):
            out_channels = base_filters * (2 ** layer_idx)
            block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2),
                nn.Dropout(p=dropout_rate)
            )
            self.conv_blocks.append(block)
            in_channels = out_channels
        
        final_spatial_size = 28 // (2 ** num_conv_layers)
        self.flat_size = in_channels * (final_spatial_size ** 2)
        
        self.fc1 = nn.Linear(self.flat_size, 50)
        self.dropout_dense = nn.Dropout(p=dropout_rate)
        self.fc2 = nn.Linear(50, 10)
        self.log_softmax = nn.LogSoftmax(dim=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.conv_blocks:
            x = block(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.dropout_dense(x)
        x = self.fc2(x)
        x = self.log_softmax(x)
        return x
    
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def load_fashion_mnist_subset(batch_size: int = 128, subset_size: float = 0.2) -> tuple:
    """Load subset of Fashion MNIST for faster experimentation"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.FashionMNIST(
        root='./data/MNIST', train=True, download=True, transform=transform
    )
    test_dataset = datasets.FashionMNIST(
        root='./data/MNIST', train=False, download=True, transform=transform
    )
    
    # Use subset for faster training
    train_indices = random.sample(range(len(train_dataset)), int(len(train_dataset) * subset_size))
    test_indices = random.sample(range(len(test_dataset)), int(len(test_dataset) * subset_size))
    
    train_subset = Subset(train_dataset, train_indices)
    test_subset = Subset(test_dataset, test_indices)
    
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader


def train_model(model: nn.Module, train_loader: DataLoader, test_loader: DataLoader,
                epochs: int = 3, lr: float = 0.01, device: str = 'cpu') -> tuple:
    """Train model and return metrics"""
    model = model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    criterion = nn.NLLLoss()
    
    start_time = time.time()
    final_train_loss = 0.0
    
    # Training
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        batch_count = 0
        
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            batch_count += 1
        
        final_train_loss = train_loss / batch_count if batch_count > 0 else 0
    
    training_time = time.time() - start_time
    
    # Testing
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    test_accuracy = 100.0 * correct / total if total > 0 else 0.0
    
    return test_accuracy, training_time, final_train_loss


def run_experiments(num_experiments: int = 30, epochs: int = 3, batch_size: int = 128,
                    output_csv: str = 'task4_results.csv', device: str = 'cpu') -> None:
    """Run automated experiments"""
    
    print("=" * 80)
    print("TASK 4: NETWORK ARCHITECTURE EXPERIMENTATION (ACCELERATED)")
    print("=" * 80)
    print(f"Dataset: Fashion MNIST (20% subset for speed)")
    print(f"Epochs: {epochs} | Batch Size: {batch_size}")
    print(f"Target Experiments: {num_experiments}")
    print("=" * 80)
    
    # Load data
    print("\nLoading Fashion MNIST...")
    train_loader, test_loader = load_fashion_mnist_subset(batch_size=batch_size, subset_size=0.2)
    print(f"✓ Loaded training and test subsets")
    
    # Define experiments
    experiments = []
    
    # Round 1: Conv Layers (hold filters=32, dropout=0.25)
    for conv_layers in [1, 2, 3, 4]:
        experiments.append({
            'conv_layers': conv_layers, 'base_filters': 32, 'dropout_rate': 0.25,
            'round': 1, 'description': f'Conv Layers: {conv_layers}'
        })
    
    # Round 2: Filters (hold conv_layers=2, dropout=0.25)
    for filters in [8, 16, 32, 64]:
        experiments.append({
            'conv_layers': 2, 'base_filters': filters, 'dropout_rate': 0.25,
            'round': 2, 'description': f'Filters: {filters}'
        })
    
    # Round 3: Dropout (hold conv_layers=2, filters=32)
    for dropout in [0.0, 0.15, 0.25, 0.40, 0.50]:
        experiments.append({
            'conv_layers': 2, 'base_filters': 32, 'dropout_rate': dropout,
            'round': 3, 'description': f'Dropout: {dropout:.2f}'
        })
    
    # Round 4: Refined combinations
    refined = [
        {'conv_layers': 2, 'base_filters': 16, 'dropout_rate': 0.25},
        {'conv_layers': 2, 'base_filters': 24, 'dropout_rate': 0.25},
        {'conv_layers': 3, 'base_filters': 16, 'dropout_rate': 0.25},
        {'conv_layers': 1, 'base_filters': 64, 'dropout_rate': 0.25},
        {'conv_layers': 2, 'base_filters': 32, 'dropout_rate': 0.20},
    ]
    for config in refined[:num_experiments - len(experiments)]:
        config['round'] = 4
        config['description'] = f"Conv: {config['conv_layers']}, F: {config['base_filters']}, D: {config['dropout_rate']}"
        experiments.append(config)
    
    experiments = experiments[:num_experiments]
    print(f"\nTotal experiments to run: {len(experiments)}\n")
    
    results = []
    
    print("=" * 80)
    print("RUNNING EXPERIMENTS")
    print("=" * 80 + "\n")
    
    for exp_idx, config in enumerate(experiments, 1):
        model = DynamicCNN(
            num_conv_layers=config['conv_layers'],
            base_filters=config['base_filters'],
            dropout_rate=config['dropout_rate']
        )
        
        num_params = model.count_parameters()
        
        print(f"[{exp_idx}/{len(experiments)}] {config['description']}")
        print(f"  Params: {num_params:,}")
        sys.stdout.flush()
        
        try:
            test_acc, train_time, final_loss = train_model(
                model, train_loader, test_loader, epochs=epochs, lr=0.01, device=device
            )
            
            result = {
                'experiment_id': exp_idx,
                'round': config['round'],
                'conv_layers': config['conv_layers'],
                'base_filters': config['base_filters'],
                'dropout_rate': config['dropout_rate'],
                'total_parameters': num_params,
                'test_accuracy': f"{test_acc:.2f}",
                'training_time_sec': f"{train_time:.2f}",
                'final_train_loss': f"{final_loss:.6f}",
                'timestamp': datetime.now().isoformat()
            }
            results.append(result)
            
            print(f"  ✓ Accuracy: {test_acc:.2f}% | Time: {train_time:.1f}s\n")
            sys.stdout.flush()
            
        except Exception as e:
            print(f"  ✗ Error: {str(e)[:50]}\n")
            sys.stdout.flush()
    
    # Save results
    if results:
        print("\n" + "=" * 80)
        print(f"SAVING RESULTS: {len(results)} experiments completed")
        print("=" * 80)
        
        with open(output_csv, 'w', newline='') as csvfile:
            fieldnames = list(results[0].keys())
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        
        print(f"✓ Results saved to: {output_csv}")
        
        # Statistics
        accuracies = [float(r['test_accuracy']) for r in results]
        times = [float(r['training_time_sec']) for r in results]
        
        print(f"\nSummary:")
        print(f"  Accuracy: min={min(accuracies):.2f}%, max={max(accuracies):.2f}%, avg={sum(accuracies)/len(accuracies):.2f}%")
        print(f"  Time: min={min(times):.1f}s, max={max(times):.1f}s, avg={sum(times)/len(times):.1f}s")
        
        best = max(results, key=lambda x: float(x['test_accuracy']))
        print(f"\nBest: Conv={best['conv_layers']}, F={best['base_filters']}, D={best['dropout_rate']} → {best['test_accuracy']}%")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Task 4: Accelerated Experimentation')
    parser.add_argument('--experiments', type=int, default=30, help='Number of experiments')
    parser.add_argument('--epochs', type=int, default=3, help='Epochs per experiment')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size')
    parser.add_argument('--output', type=str, default='task4_results.csv', help='Output CSV')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'], help='Device')
    
    args = parser.parse_args()
    device = 'cuda' if args.device == 'cuda' and torch.cuda.is_available() else 'cpu'
    
    run_experiments(
        num_experiments=args.experiments,
        epochs=args.epochs,
        batch_size=args.batch_size,
        output_csv=args.output,
        device=device
    )
