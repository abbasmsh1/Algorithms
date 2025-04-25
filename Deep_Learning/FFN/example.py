import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.datasets import load_digits, make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from models import MLP, ResidualMLP, HighwayMLP, SELU_MLP, SwishMLP, MLPWithLayerNorm, FFN_Factory
from utils import Trainer, create_dataloaders_from_numpy, set_seed


def example_classification():
    """
    Example of using the FFN models for a classification task.
    """
    print("\n=============== Classification Example ===============")
    
    # Set random seed for reproducibility
    set_seed(42)
    
    # Load dataset (digits dataset)
    digits = load_digits()
    X, y = digits.data, digits.target
    
    # Preprocess data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders_from_numpy(
        X_train, y_train, X_val, y_val, X_test, y_test, batch_size=32
    )
    
    # Define model configurations
    input_dim = X.shape[1]
    hidden_dims = [128, 64]
    output_dim = len(np.unique(y))
    
    # Create models
    models = {
        'MLP': MLP(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            batch_norm=True,
            dropout_rate=0.2
        ),
        'ResidualMLP': ResidualMLP(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            batch_norm=True,
            dropout_rate=0.2
        ),
        'SELU_MLP': SELU_MLP(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            dropout_rate=0.2
        ),
        'SwishMLP': SwishMLP(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            batch_norm=True,
            dropout_rate=0.2
        )
    }
    
    # Train and evaluate each model
    for model_name, model in models.items():
        print(f"\nTraining {model_name}...")
        
        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # Create trainer
        trainer = Trainer(model, criterion, optimizer)
        
        # Train the model
        trainer.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=20,
            early_stopping_patience=5,
            verbose=True
        )
        
        # Evaluate on test set
        test_loss, test_acc = trainer.evaluate(test_loader)
        print(f"{model_name} Test Accuracy: {test_acc:.4f}")
        
        # Plot training history
        trainer.plot_history()


def example_regression():
    """
    Example of using the FFN models for a regression task.
    """
    print("\n=============== Regression Example ===============")
    
    # Set random seed for reproducibility
    set_seed(42)
    
    # Generate synthetic regression data
    X, y = make_classification(
        n_samples=1000, 
        n_features=20, 
        n_informative=10, 
        n_redundant=5, 
        random_state=42
    )
    # Convert to regression task
    y = y.astype(np.float32) + np.random.normal(0, 0.1, size=y.shape)
    
    # Preprocess data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    # Reshape y to have shape (n_samples, 1) for MSELoss
    y_train = y_train.reshape(-1, 1)
    y_val = y_val.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders_from_numpy(
        X_train, y_train, X_val, y_val, X_test, y_test, batch_size=32
    )
    
    # Define model using the factory
    model = FFN_Factory.create_network(
        network_type='layernorm',
        input_dim=X.shape[1],
        hidden_dims=[64, 32],
        output_dim=1,
        dropout_rate=0.1
    )
    
    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Create trainer
    trainer = Trainer(model, criterion, optimizer)
    
    # Train the model
    trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=30,
        early_stopping_patience=5,
        verbose=True
    )
    
    # Evaluate on test set
    test_loss, _ = trainer.evaluate(test_loader)
    print(f"Test MSE: {test_loss:.4f}")
    
    # Plot training history
    trainer.plot_history()


def custom_model_example():
    """
    Example of building a custom FFN model with different configurations.
    """
    print("\n=============== Custom Model Example ===============")
    
    # Define model parameters
    input_dim = 10
    hidden_dims = [64, 32]
    output_dim = 5
    
    # Highway network with custom settings
    highway_model = HighwayMLP(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        output_dim=output_dim,
        activation=nn.LeakyReLU(0.1),
        dropout_rate=0.3,
        batch_norm=True,
        output_activation=nn.LogSoftmax(dim=1)
    )
    
    # Generate random input
    sample_input = torch.randn(4, input_dim)
    
    # Forward pass
    output = highway_model(sample_input)
    
    print(f"Model Architecture:\n{highway_model}")
    print(f"Input Shape: {sample_input.shape}")
    print(f"Output Shape: {output.shape}")


if __name__ == "__main__":
    # Run examples
    example_classification()
    example_regression()
    custom_model_example() 