import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional, Union, Dict, Any


class MLP(nn.Module):
    """
    Basic Multilayer Perceptron (MLP) implementation.
    
    Args:
        input_dim (int): Dimension of the input features
        hidden_dims (List[int]): List of hidden layer dimensions
        output_dim (int): Dimension of the output
        activation (nn.Module): Activation function to use (default: ReLU)
        dropout_rate (float): Dropout probability (default: 0.0)
        batch_norm (bool): Whether to use batch normalization (default: False)
        output_activation (Optional[nn.Module]): Output activation function (default: None)
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        activation: nn.Module = nn.ReLU(),
        dropout_rate: float = 0.0,
        batch_norm: bool = False,
        output_activation: Optional[nn.Module] = None,
    ):
        super(MLP, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate
        self.batch_norm = batch_norm
        
        # Build layers
        layers = []
        
        # First layer (input to first hidden)
        prev_dim = input_dim
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
                
            layers.append(activation)
            
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
                
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        # Output activation (if specified)
        if output_activation is not None:
            layers.append(output_activation)
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the MLP.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_dim)
        """
        return self.network(x)


class ResidualMLP(nn.Module):
    """
    MLP with residual connections (similar to ResNet but for fully connected layers).
    
    Args:
        input_dim (int): Dimension of the input features
        hidden_dims (List[int]): List of hidden layer dimensions
        output_dim (int): Dimension of the output
        activation (nn.Module): Activation function to use (default: ReLU)
        dropout_rate (float): Dropout probability (default: 0.0)
        batch_norm (bool): Whether to use batch normalization (default: False)
        output_activation (Optional[nn.Module]): Output activation function (default: None)
    """
    
    class ResidualBlock(nn.Module):
        """
        A residual block with two linear layers and a skip connection.
        """
        
        def __init__(
            self, 
            dim: int,
            activation: nn.Module,
            dropout_rate: float = 0.0,
            batch_norm: bool = False
        ):
            super().__init__()
            
            layers = []
            # First layer
            layers.append(nn.Linear(dim, dim))
            if batch_norm:
                layers.append(nn.BatchNorm1d(dim))
            layers.append(activation)
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
                
            # Second layer
            layers.append(nn.Linear(dim, dim))
            if batch_norm:
                layers.append(nn.BatchNorm1d(dim))
            
            self.block = nn.Sequential(*layers)
            self.activation = activation
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Forward pass with residual connection."""
            return self.activation(x + self.block(x))
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        activation: nn.Module = nn.ReLU(),
        dropout_rate: float = 0.0,
        batch_norm: bool = False,
        output_activation: Optional[nn.Module] = None,
    ):
        super(ResidualMLP, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        
        # Build layers
        layers = []
        
        # Input projection layer
        if len(hidden_dims) > 0:
            layers.append(nn.Linear(input_dim, hidden_dims[0]))
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dims[0]))
            layers.append(activation)
            
            # Residual blocks
            for i in range(len(hidden_dims)):
                dim = hidden_dims[i]
                
                # Add residual block
                layers.append(
                    self.ResidualBlock(
                        dim=dim,
                        activation=activation,
                        dropout_rate=dropout_rate,
                        batch_norm=batch_norm
                    )
                )
                
                # If next dimension is different, add projection
                if i < len(hidden_dims) - 1 and hidden_dims[i] != hidden_dims[i+1]:
                    layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
                    if batch_norm:
                        layers.append(nn.BatchNorm1d(hidden_dims[i+1]))
                    layers.append(activation)
            
            # Output layer
            layers.append(nn.Linear(hidden_dims[-1], output_dim))
        else:
            # If no hidden layers, connect input directly to output
            layers.append(nn.Linear(input_dim, output_dim))
        
        # Output activation (if specified)
        if output_activation is not None:
            layers.append(output_activation)
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the ResidualMLP.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_dim)
        """
        return self.network(x)


class HighwayMLP(nn.Module):
    """
    MLP with highway connections that allow the network to control information flow.
    
    Highway networks use gating mechanisms to decide how much of the transformation 
    or the original input should pass through at each layer.
    
    Args:
        input_dim (int): Dimension of the input features
        hidden_dims (List[int]): List of hidden layer dimensions
        output_dim (int): Dimension of the output
        activation (nn.Module): Activation function to use (default: ReLU)
        dropout_rate (float): Dropout probability (default: 0.0)
        batch_norm (bool): Whether to use batch normalization (default: False)
        output_activation (Optional[nn.Module]): Output activation function (default: None)
    """
    
    class HighwayLayer(nn.Module):
        """
        A highway layer with transform gate.
        """
        
        def __init__(
            self, 
            dim: int,
            activation: nn.Module,
            dropout_rate: float = 0.0,
            batch_norm: bool = False
        ):
            super().__init__()
            
            # Transform gate
            self.transform_gate = nn.Sequential(
                nn.Linear(dim, dim),
                nn.Sigmoid()
            )
            
            # Main transform
            layers = []
            layers.append(nn.Linear(dim, dim))
            if batch_norm:
                layers.append(nn.BatchNorm1d(dim))
            layers.append(activation)
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            
            self.transform = nn.Sequential(*layers)
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Forward pass with highway connection."""
            t = self.transform_gate(x)
            h = self.transform(x)
            return h * t + x * (1 - t)
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        activation: nn.Module = nn.ReLU(),
        dropout_rate: float = 0.0,
        batch_norm: bool = False,
        output_activation: Optional[nn.Module] = None,
    ):
        super(HighwayMLP, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        
        # Build layers
        layers = []
        
        # Input projection layer
        if len(hidden_dims) > 0:
            layers.append(nn.Linear(input_dim, hidden_dims[0]))
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dims[0]))
            layers.append(activation)
            
            # Highway layers
            for i in range(len(hidden_dims)):
                dim = hidden_dims[i]
                
                # Add highway layer
                layers.append(
                    self.HighwayLayer(
                        dim=dim,
                        activation=activation,
                        dropout_rate=dropout_rate,
                        batch_norm=batch_norm
                    )
                )
                
                # If next dimension is different, add projection
                if i < len(hidden_dims) - 1 and hidden_dims[i] != hidden_dims[i+1]:
                    layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
                    if batch_norm:
                        layers.append(nn.BatchNorm1d(hidden_dims[i+1]))
                    layers.append(activation)
            
            # Output layer
            layers.append(nn.Linear(hidden_dims[-1], output_dim))
        else:
            # If no hidden layers, connect input directly to output
            layers.append(nn.Linear(input_dim, output_dim))
        
        # Output activation (if specified)
        if output_activation is not None:
            layers.append(output_activation)
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the HighwayMLP.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_dim)
        """
        return self.network(x)


class SELU_MLP(nn.Module):
    """
    MLP with Self-Normalizing Neural Network architecture using SELU activation.
    
    SNN with SELU activation provides automatic normalization of activations, making
    it an alternative to batch normalization.
    
    Args:
        input_dim (int): Dimension of the input features
        hidden_dims (List[int]): List of hidden layer dimensions
        output_dim (int): Dimension of the output
        dropout_rate (float): Dropout probability (default: 0.0)
        output_activation (Optional[nn.Module]): Output activation function (default: None)
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        dropout_rate: float = 0.0,
        output_activation: Optional[nn.Module] = None,
    ):
        super(SELU_MLP, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate
        
        # Build layers
        layers = []
        
        # First layer (input to first hidden)
        prev_dim = input_dim
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.SELU())
            
            if dropout_rate > 0:
                layers.append(nn.AlphaDropout(dropout_rate))  # Special dropout for SELU
                
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        # Output activation (if specified)
        if output_activation is not None:
            layers.append(output_activation)
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights according to SNN recommendations
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for SELU activation using LeCun Normal initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=1/np.sqrt(m.in_features))
                nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the SELU_MLP.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_dim)
        """
        return self.network(x)


class SwishMLP(nn.Module):
    """
    MLP with Swish activation function (x * sigmoid(x)), also known as SiLU.
    
    Swish has been shown to work better than ReLU in many deep models.
    
    Args:
        input_dim (int): Dimension of the input features
        hidden_dims (List[int]): List of hidden layer dimensions
        output_dim (int): Dimension of the output
        dropout_rate (float): Dropout probability (default: 0.0)
        batch_norm (bool): Whether to use batch normalization (default: False)
        output_activation (Optional[nn.Module]): Output activation function (default: None)
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        dropout_rate: float = 0.0,
        batch_norm: bool = False,
        output_activation: Optional[nn.Module] = None,
    ):
        super(SwishMLP, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate
        self.batch_norm = batch_norm
        
        # Build layers
        layers = []
        
        # First layer (input to first hidden)
        prev_dim = input_dim
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
                
            layers.append(nn.SiLU())  # Swish activation (also known as SiLU)
            
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
                
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        # Output activation (if specified)
        if output_activation is not None:
            layers.append(output_activation)
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the SwishMLP.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_dim)
        """
        return self.network(x)


class MLPWithLayerNorm(nn.Module):
    """
    MLP with Layer Normalization instead of Batch Normalization.
    
    Layer normalization normalizes across features for each example independently,
    making it useful for tasks with variable batch sizes or for recurrent networks.
    
    Args:
        input_dim (int): Dimension of the input features
        hidden_dims (List[int]): List of hidden layer dimensions
        output_dim (int): Dimension of the output
        activation (nn.Module): Activation function to use (default: ReLU)
        dropout_rate (float): Dropout probability (default: 0.0)
        output_activation (Optional[nn.Module]): Output activation function (default: None)
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        activation: nn.Module = nn.ReLU(),
        dropout_rate: float = 0.0,
        output_activation: Optional[nn.Module] = None,
    ):
        super(MLPWithLayerNorm, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate
        
        # Build layers
        layers = []
        
        # First layer (input to first hidden)
        prev_dim = input_dim
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(activation)
            
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
                
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        # Output activation (if specified)
        if output_activation is not None:
            layers.append(output_activation)
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the MLPWithLayerNorm.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_dim)
        """
        return self.network(x)


class FFN_Factory:
    """
    Factory class for creating different types of feedforward neural networks.
    
    This class provides a convenient way to instantiate different types of FFNs
    with consistent interfaces.
    """
    
    @staticmethod
    def create_network(
        network_type: str,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        **kwargs
    ) -> nn.Module:
        """
        Create a feedforward neural network of the specified type.
        
        Args:
            network_type (str): Type of network to create
                ('mlp', 'residual', 'highway', 'selu', 'swish', 'layernorm')
            input_dim (int): Dimension of the input features
            hidden_dims (List[int]): List of hidden layer dimensions
            output_dim (int): Dimension of the output
            **kwargs: Additional arguments to pass to the network constructor
            
        Returns:
            nn.Module: The created neural network
            
        Raises:
            ValueError: If network_type is not recognized
        """
        if network_type.lower() == 'mlp':
            return MLP(input_dim, hidden_dims, output_dim, **kwargs)
        elif network_type.lower() == 'residual':
            return ResidualMLP(input_dim, hidden_dims, output_dim, **kwargs)
        elif network_type.lower() == 'highway':
            return HighwayMLP(input_dim, hidden_dims, output_dim, **kwargs)
        elif network_type.lower() == 'selu':
            return SELU_MLP(input_dim, hidden_dims, output_dim, **kwargs)
        elif network_type.lower() == 'swish':
            return SwishMLP(input_dim, hidden_dims, output_dim, **kwargs)
        elif network_type.lower() == 'layernorm':
            return MLPWithLayerNorm(input_dim, hidden_dims, output_dim, **kwargs)
        else:
            raise ValueError(f"Unknown network type: {network_type}") 