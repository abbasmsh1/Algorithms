import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Any, Optional, Union, Callable


class LeNet5(nn.Module):
    """
    Implementation of LeNet-5 CNN architecture.
    
    The LeNet-5 architecture was introduced by Yann LeCun et al. in their 1998 paper
    "Gradient-Based Learning Applied to Document Recognition". It's a simple and 
    effective CNN architecture designed for handwritten and machine-printed 
    character recognition.
    
    Args:
        num_classes (int): Number of output classes. Default is 10.
        in_channels (int): Number of input channels. Default is 1 for grayscale.
        input_size (int): Size of the input images. Default is 32.
    """
    
    def __init__(self, num_classes: int = 10, in_channels: int = 1, input_size: int = 32):
        super(LeNet5, self).__init__()
        
        # Check if input size is sufficient
        if input_size < 28:
            raise ValueError(f"Input size must be at least 28x28, got {input_size}x{input_size}")
        
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.input_size = input_size
        
        # Feature extraction part
        self.conv1 = nn.Conv2d(in_channels, 6, kernel_size=5, stride=1, padding=0)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(16, 120, kernel_size=5, stride=1, padding=0)
        
        # Calculate size after convolutions and pooling
        self.feature_size = self._calculate_conv_output_size()
        
        # Fully connected part
        self.fc1 = nn.Linear(self.feature_size, 84)
        self.fc2 = nn.Linear(84, num_classes)
        
    def _calculate_conv_output_size(self):
        """Calculate the size of the feature maps after convolutions and pooling."""
        # After conv1: size = (input_size - 5 + 0*2) + 1 = input_size - 4
        # After pool1: size = (input_size - 4) / 2
        # After conv2: size = (input_size - 4) / 2 - 4
        # After pool2: size = ((input_size - 4) / 2 - 4) / 2
        # After conv3: size = ((input_size - 4) / 2 - 4) / 2 - 4
        size = ((self.input_size - 4) // 2 - 4) // 2 - 4
        
        # If size is not sufficient, raise an error
        if size <= 0:
            raise ValueError(f"Input size {self.input_size} is too small for this architecture")
            
        return 120 * size * size
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes)
        """
        # Feature extraction
        x = self.pool1(F.tanh(self.conv1(x)))
        x = self.pool2(F.tanh(self.conv2(x)))
        x = F.tanh(self.conv3(x))
        
        # Flatten the tensor
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.tanh(self.fc1(x))
        x = self.fc2(x)
        
        return x


class AlexNet(nn.Module):
    """
    Implementation of AlexNet CNN architecture.
    
    AlexNet was introduced by Alex Krizhevsky et al. in their 2012 paper
    "ImageNet Classification with Deep Convolutional Neural Networks".
    It won the ImageNet Large Scale Visual Recognition Challenge in 2012
    and was a breakthrough in the field of computer vision.
    
    Args:
        num_classes (int): Number of output classes. Default is 1000 for ImageNet.
        in_channels (int): Number of input channels. Default is 3 for RGB.
        dropout_rate (float): Dropout probability. Default is 0.5.
    """
    
    def __init__(self, num_classes: int = 1000, in_channels: int = 3, dropout_rate: float = 0.5):
        super(AlexNet, self).__init__()
        
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.dropout_rate = dropout_rate
        
        # Feature extraction part - different from the original AlexNet due to simplification
        self.features = nn.Sequential(
            # Layer 1
            nn.Conv2d(in_channels, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            # Layer 2
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            # Layer 3
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            # Layer 4
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            # Layer 5
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        
        # Classifier part
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(256 * 6 * 6, 4096),  # Assumes input size of 224x224
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize weights using a normal distribution."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width)
                              Expected size is (batch_size, 3, 224, 224)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes)
        """
        # Feature extraction
        x = self.features(x)
        
        # Flatten the tensor
        x = x.view(x.size(0), -1)
        
        # Classification
        x = self.classifier(x)
        
        return x


class VGG16(nn.Module):
    """
    Implementation of VGG16 CNN architecture.
    
    VGG16 was introduced by Karen Simonyan and Andrew Zisserman in their 2014 paper
    "Very Deep Convolutional Networks for Large-Scale Image Recognition".
    It was a runner-up in the ImageNet Large Scale Visual Recognition Challenge in 2014.
    
    Args:
        num_classes (int): Number of output classes. Default is 1000 for ImageNet.
        in_channels (int): Number of input channels. Default is 3 for RGB.
        dropout_rate (float): Dropout probability. Default is 0.5.
        init_weights (bool): Whether to initialize weights. Default is True.
    """
    
    def __init__(self, num_classes: int = 1000, in_channels: int = 3, 
                 dropout_rate: float = 0.5, init_weights: bool = True):
        super(VGG16, self).__init__()
        
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.dropout_rate = dropout_rate
        
        # Feature extraction part
        self.features = self._make_layers(in_channels)
        
        # Classifier part
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),  # Assumes input size of 224x224
            nn.ReLU(True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(4096, num_classes),
        )
        
        # Initialize weights if specified
        if init_weights:
            self._initialize_weights()
            
    def _make_layers(self, in_channels: int) -> nn.Sequential:
        """Create the convolutional layers according to the VGG16 architecture."""
        cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
        layers = []
        
        for v in cfg:
            if v == 'M':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                layers.extend([conv2d, nn.ReLU(inplace=True)])
                in_channels = v
                
        return nn.Sequential(*layers)
        
    def _initialize_weights(self):
        """Initialize weights using a normal distribution."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width)
                              Expected size is (batch_size, 3, 224, 224)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes)
        """
        # Feature extraction
        x = self.features(x)
        
        # Flatten the tensor
        x = x.view(x.size(0), -1)
        
        # Classification
        x = self.classifier(x)
        
        return x


class ResNetBlock(nn.Module):
    """
    Basic block for ResNet architecture.
    
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        stride (int): Stride for the first convolution. Default is 1.
        downsample (nn.Module, optional): Downsampling layer. Default is None.
    """
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, 
                 downsample: Optional[nn.Module] = None):
        super(ResNetBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.downsample = downsample
        self.stride = stride
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the residual block."""
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
            
        out += identity
        out = self.relu(out)
        
        return out


class ResNet18(nn.Module):
    """
    Implementation of ResNet-18 CNN architecture.
    
    ResNet was introduced by Kaiming He et al. in their 2015 paper
    "Deep Residual Learning for Image Recognition".
    It won the ImageNet Large Scale Visual Recognition Challenge in 2015.
    
    Args:
        num_classes (int): Number of output classes. Default is 1000 for ImageNet.
        in_channels (int): Number of input channels. Default is 3 for RGB.
    """
    
    def __init__(self, num_classes: int = 1000, in_channels: int = 3):
        super(ResNet18, self).__init__()
        
        self.num_classes = num_classes
        self.in_channels = in_channels
        
        # Initial layers
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Residual layers
        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        
        # Average pooling and final fully connected layer
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        
        # Initialize weights
        self._initialize_weights()
        
    def _make_layer(self, in_channels: int, out_channels: int, blocks: int, 
                    stride: int = 1) -> nn.Sequential:
        """Create a layer with the specified number of residual blocks."""
        downsample = None
        
        # If stride > 1 or channels change, we need to downsample the identity
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
            
        layers = []
        
        # First block may have a different stride
        layers.append(ResNetBlock(in_channels, out_channels, stride, downsample))
        
        # Remaining blocks have stride 1 and no downsampling
        for _ in range(1, blocks):
            layers.append(ResNetBlock(out_channels, out_channels))
            
        return nn.Sequential(*layers)
        
    def _initialize_weights(self):
        """Initialize weights using Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width)
                              Expected size is (batch_size, 3, 224, 224)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes)
        """
        # Initial layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # Residual layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Average pooling and classification
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x


class InceptionModule(nn.Module):
    """
    Inception module for GoogLeNet architecture.
    
    Args:
        in_channels (int): Number of input channels.
        ch1x1 (int): Number of output channels for 1x1 convolution branch.
        ch3x3red (int): Number of output channels for 1x1 reduction before 3x3 convolution.
        ch3x3 (int): Number of output channels for 3x3 convolution branch.
        ch5x5red (int): Number of output channels for 1x1 reduction before 5x5 convolution.
        ch5x5 (int): Number of output channels for 5x5 convolution branch.
        pool_proj (int): Number of output channels for pool projection branch.
    """
    
    def __init__(self, in_channels: int, ch1x1: int, ch3x3red: int, ch3x3: int, 
                 ch5x5red: int, ch5x5: int, pool_proj: int):
        super(InceptionModule, self).__init__()
        
        # 1x1 convolution branch
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, ch1x1, kernel_size=1),
            nn.ReLU(inplace=True)
        )
        
        # 1x1 + 3x3 convolution branch
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, ch3x3red, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch3x3red, ch3x3, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # 1x1 + 5x5 convolution branch
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, ch5x5red, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch5x5red, ch5x5, kernel_size=5, padding=2),
            nn.ReLU(inplace=True)
        )
        
        # 3x3 pool + 1x1 convolution branch
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, pool_proj, kernel_size=1),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the inception module."""
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        
        return torch.cat([branch1, branch2, branch3, branch4], 1)


class GoogLeNet(nn.Module):
    """
    Implementation of GoogLeNet (Inception v1) CNN architecture.
    
    GoogLeNet was introduced by Christian Szegedy et al. in their 2014 paper
    "Going Deeper with Convolutions".
    It won the ImageNet Large Scale Visual Recognition Challenge in 2014.
    This implementation is a simplified version without auxiliary classifiers.
    
    Args:
        num_classes (int): Number of output classes. Default is 1000 for ImageNet.
        in_channels (int): Number of input channels. Default is 3 for RGB.
        dropout_rate (float): Dropout probability. Default is 0.4.
    """
    
    def __init__(self, num_classes: int = 1000, in_channels: int = 3, dropout_rate: float = 0.4):
        super(GoogLeNet, self).__init__()
        
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.dropout_rate = dropout_rate
        
        # Initial layers
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True)
        )
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Reduce dimension then expand
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Inception modules
        self.inception3a = InceptionModule(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = InceptionModule(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.inception4a = InceptionModule(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = InceptionModule(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = InceptionModule(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = InceptionModule(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = InceptionModule(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.inception5a = InceptionModule(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = InceptionModule(832, 384, 192, 384, 48, 128, 128)
        
        # Average pooling and final layers
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(1024, num_classes)
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize weights using Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width)
                              Expected size is (batch_size, 3, 224, 224)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes)
        """
        # Initial layers
        x = self.conv1(x)
        x = self.maxpool1(x)
        
        x = self.conv2(x)
        x = self.maxpool2(x)
        
        # Inception modules
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)
        
        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        x = self.inception4e(x)
        x = self.maxpool4(x)
        
        x = self.inception5a(x)
        x = self.inception5b(x)
        
        # Average pooling and classification
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        
        return x


class CNN_Factory:
    """
    Factory class to create various CNN architectures
    """
    @staticmethod
    def create_network(architecture, **kwargs):
        """
        Create a CNN model with the specified architecture
        
        Parameters:
        -----------
        architecture : str
            The name of the architecture to create
        **kwargs : 
            Additional arguments to pass to the model constructor
            
        Returns:
        --------
        nn.Module
            The created CNN model
        """
        architecture = architecture.lower()
        
        if architecture == 'lenet5':
            return LeNet5(**kwargs)
        elif architecture == 'alexnet':
            return AlexNet(**kwargs)
        elif architecture == 'vgg16':
            return VGG16(**kwargs)
        elif architecture == 'resnet18':
            return ResNet18(**kwargs)
        elif architecture == 'googlenet':
            return GoogLeNet(**kwargs)
        else:
            raise ValueError(f"Unknown architecture: {architecture}") 