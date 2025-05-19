# utils.py
import torch

def check_tensor(variable, name: str):
    """
    Validates that the provided variable is a torch.Tensor.
    Raises a TypeError if the check fails.
    
    Args:
        variable: The variable to check
        name (str): Name of the variable for error reporting
        
    Raises:
        TypeError: If variable is not a torch.Tensor
    """
    if not isinstance(variable, torch.Tensor):
        raise TypeError(f"The argument '{name}' must be a torch.Tensor, but got {type(variable)} instead.")

def reparameterize(mu, logvar):
    """
    Reparameterization:
    z = mu + std * eps, where eps ~ N(0, I)
    
    Args:
        mu (torch.Tensor): Mean of the latent Gaussian
        logvar (torch.Tensor): Log variance of the latent Gaussian
        
    Returns:
        torch.Tensor: Sampled latent vector
    """
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std
