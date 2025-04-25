import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional, Union, Any, Callable
from torch.utils.data import DataLoader, Dataset, TensorDataset
import torchvision.transforms as transforms
from torchvision.utils import make_grid
import torch.nn.functional as F


def calculate_conv_output_shape(input_shape: Tuple[int, int],
                               kernel_size: Union[int, Tuple[int, int]],
                               stride: Union[int, Tuple[int, int]] = 1,
                               padding: Union[int, Tuple[int, int]] = 0,
                               dilation: Union[int, Tuple[int, int]] = 1) -> Tuple[int, int]:
    """
    Calculate the output shape of a convolutional layer.
    
    Args:
        input_shape (Tuple[int, int]): Input shape (height, width).
        kernel_size (Union[int, Tuple[int, int]]): Kernel size.
        stride (Union[int, Tuple[int, int]]): Stride. Default is 1.
        padding (Union[int, Tuple[int, int]]): Padding. Default is 0.
        dilation (Union[int, Tuple[int, int]]): Dilation. Default is 1.
        
    Returns:
        Tuple[int, int]: Output shape (height, width).
    """
    # Convert to tuples if integers
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)
    
    # Calculate output shape
    h_out = int((input_shape[0] + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0] + 1)
    w_out = int((input_shape[1] + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1] + 1)
    
    return (h_out, w_out)


def calculate_pooling_output_shape(input_shape: Tuple[int, int],
                                  kernel_size: Union[int, Tuple[int, int]],
                                  stride: Optional[Union[int, Tuple[int, int]]] = None,
                                  padding: Union[int, Tuple[int, int]] = 0) -> Tuple[int, int]:
    """
    Calculate the output shape of a pooling layer.
    
    Args:
        input_shape (Tuple[int, int]): Input shape (height, width).
        kernel_size (Union[int, Tuple[int, int]]): Kernel size.
        stride (Optional[Union[int, Tuple[int, int]]]): Stride. Default is kernel_size.
        padding (Union[int, Tuple[int, int]]): Padding. Default is 0.
        
    Returns:
        Tuple[int, int]: Output shape (height, width).
    """
    # If stride is None, it defaults to kernel_size
    if stride is None:
        stride = kernel_size
    
    # Use the convolution formula (it's the same for pooling)
    return calculate_conv_output_shape(input_shape, kernel_size, stride, padding)


def visualize_filters(model: nn.Module, layer_name: str = None, 
                    fig_size: Tuple[int, int] = (12, 12), 
                    cmap: str = 'viridis') -> None:
    """
    Visualize the filters of a convolutional layer in a model.
    
    Args:
        model (nn.Module): The model containing the convolutional layer.
        layer_name (str, optional): The name of the layer to visualize.
                                   If None, visualizes the first conv layer.
        fig_size (Tuple[int, int]): Figure size. Default is (12, 12).
        cmap (str): Colormap to use. Default is 'viridis'.
    """
    # Get the first convolutional layer if layer_name is None
    conv_layer = None
    
    if layer_name is None:
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                conv_layer = module
                layer_name = name
                break
    else:
        for name, module in model.named_modules():
            if name == layer_name:
                if isinstance(module, nn.Conv2d):
                    conv_layer = module
                else:
                    raise ValueError(f"Layer {layer_name} is not a convolutional layer")
                break
    
    if conv_layer is None:
        raise ValueError("No convolutional layer found in the model")
    
    # Get the weights
    weights = conv_layer.weight.data.clone()
    
    # Normalize the weights to [0, 1] for visualization
    weights_min = weights.min()
    weights_max = weights.max()
    weights = (weights - weights_min) / (weights_max - weights_min)
    
    # Calculate grid dimensions
    n_filters = weights.shape[0]
    n_channels = weights.shape[1]
    
    # Create a grid for all filters and channels
    fig, axes = plt.subplots(n_filters, n_channels, figsize=fig_size)
    
    # Handle the case of a single filter or single channel
    if n_filters == 1 or n_channels == 1:
        axes = np.array([axes])
    
    # Reshape if both dimensions are 1
    if n_filters == 1 and n_channels == 1:
        axes = axes.reshape(1, 1)
    
    fig.suptitle(f'Filters of layer: {layer_name}', size=16)
    
    # Plot each filter's channels
    for i in range(n_filters):
        for j in range(n_channels):
            if n_filters == 1:
                ax = axes[j]
            elif n_channels == 1:
                ax = axes[i]
            else:
                ax = axes[i, j]
            
            ax.imshow(weights[i, j].cpu(), cmap=cmap)
            ax.set_xticks([])
            ax.set_yticks([])
            
            if j == 0:
                ax.set_ylabel(f'Filter {i}', size=10)
            if i == 0:
                ax.set_title(f'Channel {j}', size=10)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for the suptitle
    plt.show()


def visualize_feature_maps(model: nn.Module, image: torch.Tensor, 
                         layer_names: Optional[List[str]] = None,
                         fig_size: Tuple[int, int] = (15, 10),
                         cmap: str = 'viridis') -> None:
    """
    Visualize the feature maps generated by convolutional layers for a given image.
    
    Args:
        model (nn.Module): The model to use.
        image (torch.Tensor): Input image tensor of shape (1, C, H, W).
        layer_names (List[str], optional): List of layer names to visualize.
                                         If None, visualizes all conv layers.
        fig_size (Tuple[int, int]): Figure size. Default is (15, 10).
        cmap (str): Colormap to use. Default is 'viridis'.
    """
    # Set the model to evaluation mode
    model.eval()
    
    # Move image to the same device as the model
    device = next(model.parameters()).device
    image = image.to(device)
    
    # Create a dictionary to store the outputs of specified layers
    outputs = {}
    hooks = []
    
    # Function to capture the outputs
    def hook_fn(name):
        def hook(module, input, output):
            outputs[name] = output.detach()
        return hook
    
    # Register hooks for all convolutional layers or the specified ones
    if layer_names is None:
        layer_names = []
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                layer_names.append(name)
                hooks.append(module.register_forward_hook(hook_fn(name)))
    else:
        for name, module in model.named_modules():
            if name in layer_names:
                hooks.append(module.register_forward_hook(hook_fn(name)))
    
    # Forward pass
    with torch.no_grad():
        model(image)
    
    # Remove the hooks
    for hook in hooks:
        hook.remove()
    
    # Visualize the feature maps
    for name, feature_map in outputs.items():
        # Get the feature map for the first image in the batch
        feature_map = feature_map[0]
        
        # Calculate grid dimensions
        n_channels = feature_map.shape[0]
        rows = int(np.sqrt(n_channels))
        cols = int(np.ceil(n_channels / rows))
        
        # Create a figure
        fig, axes = plt.subplots(rows, cols, figsize=fig_size)
        fig.suptitle(f'Feature maps of layer: {name}', size=16)
        
        # Flatten axes if there's only one row or column
        if rows == 1 or cols == 1:
            axes = axes.flatten()
        
        # Plot each channel
        for i in range(n_channels):
            if rows == 1 and cols == 1:
                ax = axes
            elif rows == 1 or cols == 1:
                ax = axes[i]
            else:
                ax = axes[i // cols, i % cols]
            
            # Normalize the feature map to [0, 1]
            fm = feature_map[i].cpu()
            fm_min = fm.min()
            fm_max = fm.max()
            if fm_max > fm_min:
                fm = (fm - fm_min) / (fm_max - fm_min)
            
            ax.imshow(fm, cmap=cmap)
            ax.set_title(f'Channel {i}', size=8)
            ax.axis('off')
        
        # Hide unused axes
        for i in range(n_channels, rows * cols):
            if rows == 1 and cols == 1:
                continue
            elif rows == 1 or cols == 1:
                axes[i].axis('off')
            else:
                axes[i // cols, i % cols].axis('off')
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for the suptitle
        plt.show()


def gradcam(model: nn.Module, image: torch.Tensor, target_layer: str, 
           target_class: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Visualize Grad-CAM for the specified layer and class.
    
    Args:
        model (nn.Module): The model to use.
        image (torch.Tensor): Input image tensor of shape (1, C, H, W).
        target_layer (str): The name of the target layer for Grad-CAM.
        target_class (int, optional): The target class index.
                                    If None, uses the predicted class.
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: Tuple containing (heatmap, combined_image)
                                      as numpy arrays with values in [0, 1].
    """
    # Set the model to evaluation mode
    model.eval()
    
    # Move image to the same device as the model
    device = next(model.parameters()).device
    image = image.to(device)
    image.requires_grad_(True)
    
    # Find the target layer
    target_module = None
    for name, module in model.named_modules():
        if name == target_layer:
            target_module = module
            break
    
    if target_module is None:
        raise ValueError(f"Layer {target_layer} not found in the model")
    
    # Forward pass and get the target layer's activations
    activations = None
    def forward_hook(module, input, output):
        nonlocal activations
        activations = output
    
    hook = target_module.register_forward_hook(forward_hook)
    
    # Forward pass
    output = model(image)
    
    # If target_class is None, use the predicted class
    if target_class is None:
        target_class = output.argmax(dim=1).item()
    
    # Backward pass to get gradients
    model.zero_grad()
    one_hot = torch.zeros_like(output)
    one_hot[0, target_class] = 1
    output.backward(gradient=one_hot)
    
    # Get the gradients for the target layer
    gradients = image.grad
    
    # Remove the hook
    hook.remove()
    
    # Global average pooling of the gradients
    weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
    
    # Compute the Grad-CAM
    cam = torch.sum(weights * activations, dim=1, keepdim=True)
    cam = F.relu(cam)  # Apply ReLU to focus on features that have a positive influence
    
    # Normalize the Grad-CAM
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)
    
    # Resize the Grad-CAM to the input image size
    cam = F.interpolate(cam, size=(image.shape[2], image.shape[3]), mode='bilinear', align_corners=False)
    
    # Convert to numpy array
    cam = cam[0, 0].cpu().detach().numpy()
    
    # Convert the input image to numpy array
    input_image = image[0].permute(1, 2, 0).cpu().detach().numpy()
    input_image = (input_image - input_image.min()) / (input_image.max() - input_image.min())
    
    # Create a heatmap
    heatmap = plt.get_cmap('jet')(cam)[:, :, :3]  # Remove the alpha channel
    
    # Superimpose the heatmap on the input image
    alpha = 0.5
    combined_image = (1 - alpha) * input_image + alpha * heatmap
    
    return heatmap, combined_image


class CNNTrainer:
    """
    A trainer for convolutional neural networks.
    
    Args:
        model (nn.Module): The CNN model to train.
        criterion (nn.Module): The loss function.
        optimizer (optim.Optimizer): The optimizer.
        device (torch.device, optional): The device to use. Default is None (auto-detect).
        scheduler (optim.lr_scheduler._LRScheduler, optional): Learning rate scheduler.
    """
    
    def __init__(self, model: nn.Module, criterion: nn.Module, optimizer: optim.Optimizer,
                device: Optional[torch.device] = None,
                scheduler: Optional[optim.lr_scheduler._LRScheduler] = None):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Move model to device
        self.model.to(self.device)
        
        # History for tracking metrics
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
    
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader (DataLoader): The training data loader.
            
        Returns:
            Tuple[float, float]: Tuple containing (average_loss, accuracy).
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            # Zero the gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            # Backward pass and optimize
            loss.backward()
            self.optimizer.step()
            
            # Update statistics
            total_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
        
        # Update scheduler if provided
        if self.scheduler is not None:
            self.scheduler.step()
        
        # Return average loss and accuracy
        return total_loss / total, correct / total
    
    def evaluate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """
        Evaluate the model on the validation data.
        
        Args:
            val_loader (DataLoader): The validation data loader.
            
        Returns:
            Tuple[float, float]: Tuple containing (average_loss, accuracy).
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                # Update statistics
                total_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        
        # Return average loss and accuracy
        return total_loss / total, correct / total
    
    def fit(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None,
           epochs: int = 10, verbose: bool = True) -> Dict[str, List[float]]:
        """
        Train the model.
        
        Args:
            train_loader (DataLoader): The training data loader.
            val_loader (DataLoader, optional): The validation data loader.
            epochs (int): Number of epochs to train for. Default is 10.
            verbose (bool): Whether to print progress. Default is True.
            
        Returns:
            Dict[str, List[float]]: Dictionary containing training history.
        """
        for epoch in range(epochs):
            # Train for one epoch
            train_loss, train_acc = self.train_epoch(train_loader)
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            
            # Evaluate on validation set if provided
            if val_loader is not None:
                val_loss, val_acc = self.evaluate(val_loader)
                self.history['val_loss'].append(val_loss)
                self.history['val_acc'].append(val_acc)
                
                if verbose:
                    print(f"Epoch {epoch+1}/{epochs} - "
                         f"train_loss: {train_loss:.4f}, train_acc: {train_acc:.4f}, "
                         f"val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}")
            else:
                if verbose:
                    print(f"Epoch {epoch+1}/{epochs} - "
                         f"train_loss: {train_loss:.4f}, train_acc: {train_acc:.4f}")
        
        return self.history
    
    def predict(self, test_loader: DataLoader) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make predictions on the test data.
        
        Args:
            test_loader (DataLoader): The test data loader.
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple containing (predictions, probabilities).
        """
        self.model.eval()
        all_preds = []
        all_probs = []
        
        with torch.no_grad():
            for inputs, _ in test_loader:
                inputs = inputs.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                
                # Get predictions and probabilities
                _, preds = torch.max(outputs, 1)
                probs = F.softmax(outputs, dim=1)
                
                all_preds.append(preds.cpu())
                all_probs.append(probs.cpu())
        
        # Concatenate all predictions and probabilities
        all_preds = torch.cat(all_preds)
        all_probs = torch.cat(all_probs)
        
        return all_preds, all_probs
    
    def plot_history(self) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot the training history.
        
        Returns:
            Tuple[plt.Figure, plt.Axes]: Tuple containing (figure, axes).
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot loss
        axes[0].plot(self.history['train_loss'], label='Training Loss')
        if len(self.history['val_loss']) > 0:
            axes[0].plot(self.history['val_loss'], label='Validation Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Loss per Epoch')
        axes[0].legend()
        
        # Plot accuracy
        axes[1].plot(self.history['train_acc'], label='Training Accuracy')
        if len(self.history['val_acc']) > 0:
            axes[1].plot(self.history['val_acc'], label='Validation Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Accuracy per Epoch')
        axes[1].legend()
        
        plt.tight_layout()
        plt.show()
        
        return fig, axes
    
    def save_model(self, path: str) -> None:
        """
        Save the model.
        
        Args:
            path (str): Path to save the model.
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
        }, path)
    
    def load_model(self, path: str) -> None:
        """
        Load the model.
        
        Args:
            path (str): Path to load the model from.
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint['history'] 