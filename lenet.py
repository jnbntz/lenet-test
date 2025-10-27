#!/usr/bin/env python3
"""
LeNet-5 Neural Network Implementation
Classic CNN architecture for digit recognition on MNIST dataset
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime


class LeNet5(nn.Module):
    """
    LeNet-5 Convolutional Neural Network
    Original architecture by Yann LeCun et al. (1998)
    """
    def __init__(self):
        super(LeNet5, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)  # Input: 1x28x28, Output: 6x28x28
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)  # Input: 6x14x14, Output: 16x10x10
        
        # Fully connected layers
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # After pooling: 16x5x5
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)  # 10 classes for digits 0-9
        
    def forward(self, x):
        # Layer 1: Conv -> ReLU -> MaxPool
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        
        # Layer 2: Conv -> ReLU -> MaxPool
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        
        # Flatten
        x = x.view(-1, 16 * 5 * 5)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x


def download_and_prepare_data(batch_size=64):
    """
    Download MNIST dataset and create data loaders
    
    Args:
        batch_size: Batch size for training and testing
        
    Returns:
        train_loader, test_loader
    """
    print("Downloading and preparing MNIST dataset...")
    
    # Define transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
    ])
    
    # Download and load training data
    train_dataset = datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )
    
    # Download and load test data
    test_dataset = datasets.MNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    return train_loader, test_loader


def train(model, device, train_loader, optimizer, criterion, epoch):
    """
    Train the model for one epoch
    
    Args:
        model: Neural network model
        device: Device to train on (CPU or CUDA)
        train_loader: Training data loader
        optimizer: Optimization algorithm
        criterion: Loss function
        epoch: Current epoch number
        
    Returns:
        Average training loss
    """
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        output = model(data)
        loss = criterion(output, target)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Calculate statistics
        train_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        
        # Print progress
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
    
    avg_loss = train_loss / len(train_loader)
    accuracy = 100. * correct / total
    print(f'Epoch {epoch} - Average Loss: {avg_loss:.6f}, Accuracy: {accuracy:.2f}%')
    
    return avg_loss, accuracy


def test(model, device, test_loader, criterion):
    """
    Evaluate the model on test data
    
    Args:
        model: Neural network model
        device: Device to test on (CPU or CUDA)
        test_loader: Test data loader
        criterion: Loss function
        
    Returns:
        Test loss and accuracy
    """
    model.eval()
    test_loss = 0
    correct = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    test_loss /= len(test_loader)
    accuracy = 100. * correct / len(test_loader.dataset)
    
    print(f'\nTest set: Average loss: {test_loss:.4f}, '
          f'Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n')
    
    return test_loss, accuracy


def visualize_predictions(model, device, test_loader, num_images=10):
    """
    Visualize model predictions on sample images
    
    Args:
        model: Trained model
        device: Device model is on
        test_loader: Test data loader
        num_images: Number of images to display
    """
    model.eval()
    
    # Get a batch of test data
    data_iter = iter(test_loader)
    images, labels = next(data_iter)
    
    # Make predictions
    images_to_show = images[:num_images]
    labels_to_show = labels[:num_images]
    
    with torch.no_grad():
        outputs = model(images_to_show.to(device))
        _, predictions = torch.max(outputs, 1)
    
    # Plot images with predictions
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    axes = axes.ravel()
    
    for i in range(num_images):
        img = images_to_show[i].cpu().numpy().squeeze()
        axes[i].imshow(img, cmap='gray')
        
        pred = predictions[i].cpu().item()
        true_label = labels_to_show[i].item()
        color = 'green' if pred == true_label else 'red'
        
        axes[i].set_title(f'Pred: {pred}, True: {true_label}', color=color)
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('predictions.png')
    print("Predictions visualization saved as 'predictions.png'")


def plot_training_history(train_losses, train_accs, test_losses, test_accs):
    """
    Plot training and testing metrics
    
    Args:
        train_losses: List of training losses per epoch
        train_accs: List of training accuracies per epoch
        test_losses: List of test losses per epoch
        test_accs: List of test accuracies per epoch
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    epochs = range(1, len(train_losses) + 1)
    
    # Plot losses
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss')
    ax1.plot(epochs, test_losses, 'r-', label='Test Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Test Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracies
    ax2.plot(epochs, train_accs, 'b-', label='Training Accuracy')
    ax2.plot(epochs, test_accs, 'r-', label='Test Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Test Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    print("Training history saved as 'training_history.png'")


def main():
    """
    Main function to run the complete training pipeline
    """
    # Hyperparameters
    batch_size = 64
    learning_rate = 0.001
    epochs = 10
    
    print("=" * 60)
    print("LeNet-5 Neural Network on MNIST Dataset")
    print("=" * 60)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print()
    
    # Download and prepare data
    train_loader, test_loader = download_and_prepare_data(batch_size)
    print()
    
    # Initialize model
    model = LeNet5().to(device)
    print("Model architecture:")
    print(model)
    print()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print()
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    train_losses = []
    train_accs = []
    test_losses = []
    test_accs = []
    
    print("Starting training...")
    print("=" * 60)
    
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train(model, device, train_loader, optimizer, criterion, epoch)
        test_loss, test_acc = test(model, device, test_loader, criterion)
        
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)
    
    print("=" * 60)
    print("Training completed!")
    print()
    
    # Save the model
    model_path = 'lenet5_mnist.pth'
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to '{model_path}'")
    print()
    
    # Plot training history
    plot_training_history(train_losses, train_accs, test_losses, test_accs)
    
    # Visualize some predictions
    visualize_predictions(model, device, test_loader)
    
    print()
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)


if __name__ == "__main__":
    main()

