import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader

# Import our modules
from models import CNN_Factory
from utils import CNNTrainer, visualize_filters, visualize_feature_maps, gradcam


def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Define transformations for CIFAR-10
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    
    # Load CIFAR-10 dataset
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    trainloader = DataLoader(
        trainset, batch_size=128, shuffle=True, num_workers=2)
    
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = DataLoader(
        testset, batch_size=128, shuffle=False, num_workers=2)
    
    # Define class names for CIFAR-10
    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')
    
    # Example 1: Train a LeNet model
    print("\nExample 1: Training LeNet5 on CIFAR-10")
    train_lenet(trainloader, testloader, device)
    
    # Example 2: Visualize filters and feature maps of a pre-trained model
    print("\nExample 2: Visualizing filters and feature maps of AlexNet")
    visualize_alexnet(testset, device)
    
    # Example 3: Apply Grad-CAM to a pre-trained model
    print("\nExample 3: Applying Grad-CAM to ResNet18")
    apply_gradcam(testset, classes, device)


def train_lenet(trainloader, testloader, device):
    # Create LeNet5 model
    model = CNN_Factory.create_network('lenet5', num_classes=10, in_channels=3, input_size=(32, 32))
    model = model.to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    
    # Create trainer
    trainer = CNNTrainer(model, criterion, optimizer, device, scheduler)
    
    # Train for a small number of epochs (for demonstration)
    print("Training LeNet5 for 2 epochs...")
    trainer.fit(trainloader, testloader, epochs=2)
    
    # Plot training history
    trainer.plot_history()
    
    # Evaluate on test set
    test_loss, test_acc = trainer.evaluate(testloader)
    print(f"Test accuracy: {test_acc:.4f}")
    
    # Save the model
    trainer.save_model('lenet5_cifar10.pth')
    print("Model saved to lenet5_cifar10.pth")


def visualize_alexnet(testset, device):
    # Create AlexNet model
    model = CNN_Factory.create_network('alexnet', num_classes=10, in_channels=3, input_size=(224, 224))
    model = model.to(device)
    
    # Get a sample image and resize it to match AlexNet's input size
    sample_idx = 25  # Choose an example
    image, label = testset[sample_idx]
    image = transforms.Resize((224, 224))(image)
    image_batch = image.unsqueeze(0).to(device)  # Add batch dimension
    
    # Visualize filters
    print("Visualizing filters of the first convolutional layer...")
    visualize_filters(model, 'features.0', fig_size=(8, 8))
    
    # Visualize feature maps
    print("Visualizing feature maps...")
    visualize_feature_maps(model, image_batch, ['features.0', 'features.3', 'features.6'], fig_size=(10, 6))


def apply_gradcam(testset, classes, device):
    # Load a pre-trained ResNet18 model
    model = torchvision.models.resnet18(pretrained=True)
    model = model.to(device)
    model.eval()
    
    # Get a sample image
    sample_idx = 12  # Choose an example
    image, label = testset[sample_idx]
    
    # Resize to match ResNet input size and normalize
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = transform(transforms.ToPILImage()(image))
    image_batch = image.unsqueeze(0).to(device)  # Add batch dimension
    
    # Apply Grad-CAM
    print(f"Applying Grad-CAM for class: {classes[label]}")
    heatmap, combined = gradcam(model, image_batch, target_layer='layer4.1.conv2', target_class=label)
    
    # Display the original image, heatmap, and combined view
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image (convert back to 0-1 range)
    orig_img = image.permute(1, 2, 0).cpu().numpy()
    orig_img = (orig_img - orig_img.min()) / (orig_img.max() - orig_img.min())
    axes[0].imshow(orig_img)
    axes[0].set_title(f'Original ({classes[label]})')
    axes[0].axis('off')
    
    # Heatmap
    axes[1].imshow(heatmap)
    axes[1].set_title('Grad-CAM Heatmap')
    axes[1].axis('off')
    
    # Combined
    axes[2].imshow(combined)
    axes[2].set_title('Grad-CAM Overlay')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main() 