"""
Train a ResNet-18 model on CIFAR-10.
Saves the trained checkpoint for adversarial evaluation.
"""

import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models


def get_cifar10_loaders(config):
    """Load CIFAR-10 train and test sets."""
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    data_dir = config["data"]["data_dir"]
    batch_size = config["data"]["batch_size"]
    num_workers = config["data"]["num_workers"]

    train_set = datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform_train
    )
    test_set = datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform_test
    )

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, test_loader


def build_resnet18(num_classes=10):
    """Build a ResNet-18 adapted for CIFAR-10 (32x32 images)."""
    model = models.resnet18(weights=None, num_classes=num_classes)
    # Adapt first conv layer for 32x32 input (no aggressive downsampling)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()  # Remove maxpool for small images
    return model


def train(model, train_loader, device, epochs=20, lr=0.01):
    """Train the model with SGD + cosine annealing."""
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9,
                          weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        scheduler.step()
        acc = 100.0 * correct / total
        print(f"Epoch {epoch+1}/{epochs} | Loss: {running_loss/len(train_loader):.4f} | Acc: {acc:.2f}%")

    return model


def evaluate(model, test_loader, device):
    """Evaluate clean accuracy on test set."""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    acc = 100.0 * correct / total
    print(f"Clean Test Accuracy: {acc:.2f}%")
    return acc


def main():
    # Load config
    with open("configs/default_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    train_loader, test_loader = get_cifar10_loaders(config)

    # Build and train model
    model = build_resnet18(num_classes=config["model"]["num_classes"])
    model = train(model, train_loader, device, epochs=20)

    # Evaluate
    evaluate(model, test_loader, device)

    # Save checkpoint
    os.makedirs("outputs", exist_ok=True)
    checkpoint_path = config["model"]["checkpoint_path"]
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Model saved to {checkpoint_path}")


if __name__ == "__main__":
    main()
