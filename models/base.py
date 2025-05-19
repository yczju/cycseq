# base.py
import torch
from torch import nn

class View(nn.Module):
    """
    A module that reshapes an input tensor to a specified shape.
    Useful for connecting layers with different dimensional requirements.
    """
    def __init__(self, *size):
        """
        Initialize the View module.

        Args:
            *size: A sequence of integers defining the desired shape.
        """
        super(View, self).__init__()
        self.size = size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reshape the input tensor.

        Args:
            x (torch.Tensor): Tensor to be reshaped.

        Returns:
            torch.Tensor: Reshaped tensor.
            
        Raises:
            RuntimeError: If reshaping fails
        """
        check_tensor(x, 'x')
        try:
            return x.view(*self.size)
        except Exception as error:
            raise RuntimeError(f"Error reshaping tensor with view size {self.size}: {error}")

class _Sampler(nn.Module):
    """
    A module that implements the reparameterization trick for VAEs.
    Samples from a Gaussian distribution with the given mean and log variance.
    """
    def __init__(self):
        super(_Sampler, self).__init__()
    
    def forward(self, mean: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """
        Sample from a Gaussian distribution using the reparameterization trick.
        
        Args:
            mean (torch.Tensor): Mean of the Gaussian distribution
            log_var (torch.Tensor): Log variance of the Gaussian distribution
            
        Returns:
            torch.Tensor: Sampled latent vector
            
        Raises:
            RuntimeError: If sampling fails
        """
        check_tensor(mean, 'mean')
        check_tensor(log_var, 'log_var')
        try:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            z = mean + eps * std
            return z
        except Exception as error:
            raise RuntimeError(f"Error during sampling: {error}")

class ConditionEmbedding(nn.Module):
    """
    Maps categorical condition indices to dense vector embeddings.
    Used for representing categorical variables like batch or perturbation.
    """
    def __init__(self, num_embeddings, embedding_dim):
        """
        Initialize the condition embedding module.
        
        Args:
            num_embeddings (int): Number of distinct categories to embed
            embedding_dim (int): Dimension of the embedding vectors
        """
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        
    def forward(self, x):
        """
        Convert indices to embeddings.
        
        Args:
            x (torch.Tensor): Tensor of indices to embed
            
        Returns:
            torch.Tensor: Tensor of embeddings
        """
        return self.embedding(x)
