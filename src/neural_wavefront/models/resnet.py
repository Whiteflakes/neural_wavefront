"""
ResNet-based model for Zernike coefficient regression.

This module implements a CNN architecture based on ResNet-18 for predicting
Zernike coefficients from PSF images.
"""

import torch
import torch.nn as nn
from torchvision import models
from typing import Optional


class ZernikeRegressor(nn.Module):
    """
    ResNet-based regressor for predicting Zernike coefficients from PSF images.
    
    The model uses a ResNet-18 backbone modified for single-channel input
    (grayscale PSF images) and regression output (Zernike coefficients).
    
    Args:
        n_modes: Number of Zernike coefficients to predict
        pretrained: Whether to use ImageNet pretrained weights (default: False)
        dropout: Dropout rate for regularization (default: 0.2)
    
    Input:
        x: Tensor of shape (batch_size, 1, H, W) - grayscale PSF images
    
    Output:
        Tensor of shape (batch_size, n_modes) - predicted Zernike coefficients
    
    Example:
        >>> model = ZernikeRegressor(n_modes=15, pretrained=False, dropout=0.2)
        >>> x = torch.randn(32, 1, 256, 256)  # Batch of 32 PSF images
        >>> coeffs = model(x)
        >>> coeffs.shape
        torch.Size([32, 15])
    """
    
    def __init__(
        self,
        n_modes: int = 15,
        pretrained: bool = False,
        dropout: float = 0.2
    ):
        super(ZernikeRegressor, self).__init__()
        
        self.n_modes = n_modes
        self.dropout_rate = dropout
        
        # Load ResNet-18 backbone
        if pretrained:
            print("Loading pretrained ResNet-18 weights from ImageNet...")
            self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        else:
            print("Initializing ResNet-18 with random weights...")
            self.backbone = models.resnet18(weights=None)
        
        # Modify first conv layer for single-channel input
        # Original: Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        # New: Conv2d(1, 64, kernel_size=7, stride=2, padding=3)
        original_conv1 = self.backbone.conv1
        self.backbone.conv1 = nn.Conv2d(
            in_channels=1,  # Single channel for grayscale PSF
            out_channels=original_conv1.out_channels,
            kernel_size=original_conv1.kernel_size,
            stride=original_conv1.stride,
            padding=original_conv1.padding,
            bias=False
        )
        
        # If pretrained, average the RGB weights to initialize single-channel conv
        if pretrained:
            with torch.no_grad():
                # Average the 3-channel weights to 1-channel
                self.backbone.conv1.weight = nn.Parameter(
                    original_conv1.weight.mean(dim=1, keepdim=True)
                )
        
        # Replace final FC layer for regression
        # ResNet-18 has 512 features before the final FC layer
        in_features = self.backbone.fc.in_features
        
        # Create new regression head
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(256, n_modes)  # Output: Zernike coefficients
        )
        
        print(f"✅ Model initialized: {self.count_parameters():,} parameters")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input PSF images, shape (batch_size, 1, H, W)
        
        Returns:
            Predicted Zernike coefficients, shape (batch_size, n_modes)
        """
        return self.backbone(x)
    
    def count_parameters(self) -> int:
        """Count the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_feature_maps(self, x: torch.Tensor, layer_name: str = 'layer4') -> torch.Tensor:
        """
        Extract intermediate feature maps for visualization.
        
        Args:
            x: Input PSF images
            layer_name: Name of layer to extract ('layer1', 'layer2', 'layer3', 'layer4')
        
        Returns:
            Feature maps from the specified layer
        """
        features = {}
        
        def hook_fn(module, input, output):
            features['output'] = output
        
        # Register hook
        target_layer = getattr(self.backbone, layer_name)
        handle = target_layer.register_forward_hook(hook_fn)
        
        # Forward pass
        _ = self.forward(x)
        
        # Remove hook
        handle.remove()
        
        return features['output']


class LightweightRegressor(nn.Module):
    """
    Lightweight CNN for faster training and testing.
    
    This is a smaller model for quick prototyping or resource-constrained scenarios.
    
    Args:
        n_modes: Number of Zernike coefficients to predict
        dropout: Dropout rate (default: 0.2)
    """
    
    def __init__(self, n_modes: int = 15, dropout: float = 0.2):
        super(LightweightRegressor, self).__init__()
        
        self.n_modes = n_modes
        
        # Simple CNN architecture
        self.features = nn.Sequential(
            # Conv block 1: 1 -> 32
            nn.Conv2d(1, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Conv block 2: 32 -> 64
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Conv block 3: 64 -> 128
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Conv block 4: 128 -> 256
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Regression head
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(128, n_modes)
        )
        
        print(f"✅ Lightweight model initialized: {self.count_parameters():,} parameters")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        features = self.features(x)
        coeffs = self.regressor(features)
        return coeffs
    
    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_model(
    model_name: str = "ResNet18",
    n_modes: int = 15,
    pretrained: bool = False,
    dropout: float = 0.2,
    **kwargs
) -> nn.Module:
    """
    Factory function to create model by name.
    
    Args:
        model_name: Model architecture name ('ResNet18', 'Lightweight')
        n_modes: Number of Zernike coefficients
        pretrained: Use pretrained weights (only for ResNet18)
        dropout: Dropout rate
        **kwargs: Additional model-specific arguments
    
    Returns:
        Initialized model
    
    Example:
        >>> model = create_model('ResNet18', n_modes=15, pretrained=False)
    """
    model_name = model_name.lower()
    
    if model_name == 'resnet18':
        return ZernikeRegressor(
            n_modes=n_modes,
            pretrained=pretrained,
            dropout=dropout
        )
    elif model_name == 'lightweight':
        return LightweightRegressor(
            n_modes=n_modes,
            dropout=dropout
        )
    else:
        raise ValueError(f"Unknown model: {model_name}. Choose 'ResNet18' or 'Lightweight'")


# Demo/test code
if __name__ == "__main__":
    print("="*60)
    print("Model Architecture Demonstration")
    print("="*60)
    
    # Test ResNet-18 model
    print("\n1. Testing ResNet-18 model...")
    model_resnet = ZernikeRegressor(n_modes=15, pretrained=False, dropout=0.2)
    
    # Create dummy input
    batch_size = 8
    x = torch.randn(batch_size, 1, 256, 256)
    
    # Forward pass
    print(f"   Input shape: {x.shape}")
    with torch.no_grad():
        output = model_resnet(x)
    print(f"   Output shape: {output.shape}")
    print(f"   ✅ ResNet-18 forward pass successful")
    
    # Test lightweight model
    print("\n2. Testing Lightweight model...")
    model_light = LightweightRegressor(n_modes=15, dropout=0.2)
    
    with torch.no_grad():
        output_light = model_light(x)
    print(f"   Output shape: {output_light.shape}")
    print(f"   ✅ Lightweight forward pass successful")
    
    # Compare model sizes
    print("\n3. Model comparison:")
    print(f"   ResNet-18: {model_resnet.count_parameters():,} parameters")
    print(f"   Lightweight: {model_light.count_parameters():,} parameters")
    
    # Test factory function
    print("\n4. Testing factory function...")
    model_factory = create_model('ResNet18', n_modes=15)
    print(f"   ✅ Factory function working")
    
    # Test feature extraction
    print("\n5. Testing feature extraction...")
    features = model_resnet.get_feature_maps(x[:2], layer_name='layer4')
    print(f"   Feature map shape: {features.shape}")
    print(f"   ✅ Feature extraction working")
    
    print("\n" + "="*60)
    print("✅ Model architecture implementation complete!")
    print("="*60)
