import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Tuple, Callable, Union, Optional
from torch.utils.data import DataLoader, TensorDataset


class Trainer:
    """
    A trainer for feedforward neural networks.
    
    Handles the training loop, evaluation, and tracking of metrics.
    
    Args:
        model (nn.Module): The neural network model to train
        criterion (nn.Module): Loss function
        optimizer (optim.Optimizer): Optimizer
        device (torch.device): Device to use for training (CPU or GPU)
    """
    
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        device: torch.device = None,
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # History for tracking metrics
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
        }
    
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """
        Train the model for one epoch.
        
        Args:
            train_loader (DataLoader): DataLoader for training data
            
        Returns:
            Tuple[float, float]: Average loss and accuracy for the epoch
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
            
            # Calculate metrics
            total_loss += loss.item() * inputs.size(0)
            
            # For classification tasks
            if len(targets.shape) == 1 or targets.shape[1] == 1:
                _, predicted = torch.max(outputs, 1)
                if len(targets.shape) > 1 and targets.shape[1] == 1:
                    targets = targets.squeeze()
                correct += (predicted == targets).sum().item()
                total += targets.size(0)
        
        # Calculate average metrics
        avg_loss = total_loss / len(train_loader.dataset)
        avg_acc = correct / total if total > 0 else 0.0
        
        return avg_loss, avg_acc
    
    def evaluate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """
        Evaluate the model on validation data.
        
        Args:
            val_loader (DataLoader): DataLoader for validation data
            
        Returns:
            Tuple[float, float]: Average loss and accuracy
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
                
                # Calculate metrics
                total_loss += loss.item() * inputs.size(0)
                
                # For classification tasks
                if len(targets.shape) == 1 or targets.shape[1] == 1:
                    _, predicted = torch.max(outputs, 1)
                    if len(targets.shape) > 1 and targets.shape[1] == 1:
                        targets = targets.squeeze()
                    correct += (predicted == targets).sum().item()
                    total += targets.size(0)
        
        # Calculate average metrics
        avg_loss = total_loss / len(val_loader.dataset)
        avg_acc = correct / total if total > 0 else 0.0
        
        return avg_loss, avg_acc
    
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 10,
        early_stopping_patience: int = 0,
        verbose: bool = True,
    ) -> Dict[str, List[float]]:
        """
        Train the model for multiple epochs.
        
        Args:
            train_loader (DataLoader): DataLoader for training data
            val_loader (Optional[DataLoader]): DataLoader for validation data
            epochs (int): Number of epochs to train for
            early_stopping_patience (int): Number of epochs with no improvement 
                                           to wait before stopping
            verbose (bool): Whether to print progress
            
        Returns:
            Dict[str, List[float]]: Training history
        """
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Train one epoch
            train_loss, train_acc = self.train_epoch(train_loader)
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            
            # Evaluate on validation set if provided
            if val_loader is not None:
                val_loss, val_acc = self.evaluate(val_loader)
                self.history['val_loss'].append(val_loss)
                self.history['val_acc'].append(val_acc)
                
                # Early stopping
                if early_stopping_patience > 0:
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        if patience_counter >= early_stopping_patience:
                            if verbose:
                                print(f"Early stopping at epoch {epoch+1}")
                            break
                
                if verbose:
                    print(f"Epoch {epoch+1}/{epochs} - "
                          f"train_loss: {train_loss:.4f}, train_acc: {train_acc:.4f}, "
                          f"val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}")
            else:
                if verbose:
                    print(f"Epoch {epoch+1}/{epochs} - "
                          f"train_loss: {train_loss:.4f}, train_acc: {train_acc:.4f}")
        
        return self.history
    
    def predict(self, test_loader: DataLoader) -> torch.Tensor:
        """
        Make predictions with the model.
        
        Args:
            test_loader (DataLoader): DataLoader for test data
            
        Returns:
            torch.Tensor: Predictions
        """
        self.model.eval()
        all_outputs = []
        
        with torch.no_grad():
            for inputs, _ in test_loader:
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                all_outputs.append(outputs.cpu())
        
        return torch.cat(all_outputs, dim=0)
    
    def plot_history(self) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot the training history.
        
        Returns:
            Tuple[plt.Figure, plt.Axes]: Figure and axes objects
        """
        fig, axs = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot loss
        axs[0].plot(self.history['train_loss'], label='Training Loss')
        if self.history['val_loss']:
            axs[0].plot(self.history['val_loss'], label='Validation Loss')
        axs[0].set_xlabel('Epoch')
        axs[0].set_ylabel('Loss')
        axs[0].set_title('Loss Over Time')
        axs[0].legend()
        axs[0].grid(True)
        
        # Plot accuracy
        axs[1].plot(self.history['train_acc'], label='Training Accuracy')
        if self.history['val_acc']:
            axs[1].plot(self.history['val_acc'], label='Validation Accuracy')
        axs[1].set_xlabel('Epoch')
        axs[1].set_ylabel('Accuracy')
        axs[1].set_title('Accuracy Over Time')
        axs[1].legend()
        axs[1].grid(True)
        
        plt.tight_layout()
        return fig, axs


def create_dataloaders_from_numpy(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    X_test: Optional[np.ndarray] = None,
    y_test: Optional[np.ndarray] = None,
    batch_size: int = 32,
    num_workers: int = 0,
) -> Tuple[DataLoader, Optional[DataLoader], Optional[DataLoader]]:
    """
    Create PyTorch DataLoaders from NumPy arrays.
    
    Args:
        X_train (np.ndarray): Training features
        y_train (np.ndarray): Training targets
        X_val (np.ndarray, optional): Validation features
        y_val (np.ndarray, optional): Validation targets
        X_test (np.ndarray, optional): Test features
        y_test (np.ndarray, optional): Test targets
        batch_size (int): Batch size
        num_workers (int): Number of workers for DataLoader
        
    Returns:
        Tuple: (train_loader, val_loader, test_loader)
    """
    # Convert to torch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long if y_train.dtype == np.int64 else torch.float32)
    
    # Create datasets
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers
    )
    
    val_loader = None
    if X_val is not None and y_val is not None:
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val, dtype=torch.long if y_val.dtype == np.int64 else torch.float32)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=num_workers
        )
    
    test_loader = None
    if X_test is not None and y_test is not None:
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test, dtype=torch.long if y_test.dtype == np.int64 else torch.float32)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        test_loader = DataLoader(
            test_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=num_workers
        )
    
    return train_loader, val_loader, test_loader


def set_seed(seed: int) -> None:
    """
    Set the random seed for reproducibility.
    
    Args:
        seed (int): Random seed
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False 