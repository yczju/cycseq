# attention.py
import torch
from torch import nn
from ..util.utils import check_tensor

class SelfAttention(nn.Module):
    """
    Single-head self-attention mechanism that allows the model to focus on 
    different parts of the input based on their relevance.
    """
    def __init__(self, in_dim: int):
        """
        Initialize the self-attention module.
        
        Args:
            in_dim (int): Dimension of the input features
        """
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Linear(in_dim, in_dim)
        self.key_conv = nn.Linear(in_dim, in_dim)
        self.value_conv = nn.Linear(in_dim, in_dim)
        self.gamma = nn.Parameter(torch.zeros(1))  # Learnable scaling parameter
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply self-attention on the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_dim).
            
        Returns:
            torch.Tensor: Attention-weighted output of shape (batch_size, in_dim)
            
        Raises:
            RuntimeError: If attention computation fails
        """
        check_tensor(x, 'x')
        try:
            # We treat each sample as a "sequence" of length=1 for demonstration.
            x_unsqueezed = x.unsqueeze(1)  # (batch_size, 1, in_dim)
            
            # Compute Q, K, V
            queries = self.query_conv(x_unsqueezed)  # (batch_size, 1, in_dim)
            keys = self.key_conv(x_unsqueezed)       # (batch_size, 1, in_dim)
            values = self.value_conv(x_unsqueezed)   # (batch_size, 1, in_dim)
            
            # Compute attention scores
            attention_scores = torch.bmm(queries, keys.transpose(1, 2))  # (batch_size, 1, 1)
            attention_weights = self.softmax(attention_scores)           # normalize scores
            
            # Weighted sum of values
            attention_output = torch.bmm(attention_weights, values)      # (batch_size, 1, in_dim)
            
            # Fuse
            out = self.gamma * attention_output + x_unsqueezed
            return out.squeeze(1)  # back to (batch_size, in_dim)
        except Exception as error:
            raise RuntimeError(f"Error in SelfAttention forward pass: {error}")

