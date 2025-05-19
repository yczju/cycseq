# residual.py
import torch
from torch import nn
from .utils import check_tensor

class ResidualBlock(nn.Module):
    """
    Residual block with batch normalization, dropout, and PReLU activation.
    Helps with gradient flow in deep networks by adding skip connections.
    """
    def __init__(self, in_features: int, dropout_rate=0.2):
        """
        Initialize a residual block.
        
        Args:
            in_features (int): Number of input features
            dropout_rate (float): Dropout probability, default 0.2
        """
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(in_features, in_features)
        self.bn1 = nn.BatchNorm1d(in_features)
        self.prelu1 = nn.PReLU()
        self.drop1 = nn.Dropout(dropout_rate)

        self.fc2 = nn.Linear(in_features, in_features)
        self.bn2 = nn.BatchNorm1d(in_features)
        self.prelu2 = nn.PReLU()
        self.drop2 = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the residual block.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Output tensor with same shape as input
            
        Raises:
            RuntimeError: If forward pass fails
        """
        check_tensor(x, 'x')
        try:
            residual = x
            # First transformation
            out = self.fc1(x)
            out = self.bn1(out)
            out = self.prelu1(out)
            out = self.drop1(out)

            # Second transformation
            out = self.fc2(out)
            out = self.bn2(out)
            # residual connection
            out += residual
            out = self.prelu2(out)
            out = self.drop2(out)

            return out
        except Exception as error:
            raise RuntimeError(f"Error in ResidualBlock forward pass: {error}")

