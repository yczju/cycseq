# metrics.py
import torch
from scipy import stats
import torch.nn as nn
import torch.nn.functional as F

def r2_loss(x, y):
    """
    Calculate the R-squared loss between predicted values x and target values y.
    
    This function computes the coefficient of determination (R²) for each sample
    in the batch and returns 1 minus the average R² as a loss value. Lower values
    indicate better fit between predictions and targets.
    
    Parameters:
        x (torch.Tensor or numpy.ndarray): Predicted values with shape (batch_size, n_features)
        y (torch.Tensor or numpy.ndarray): Target values with shape (batch_size, n_features)
        
    Returns:
        torch.Tensor: A scalar tensor containing the R² loss (1 - avg_r2_score),
                     moved to GPU if available
    
    Note:
        The R² score measures the proportion of variance in the dependent variable
        that is predictable from the independent variable. The loss is defined as
        1 minus this value so that minimizing the loss corresponds to maximizing R².
    """
    batch_size = x.shape[0]
    r2_scores_sum = 0.0

    for i in range(batch_size):
        _, _, r_value, _, _ = stats.linregress(x[i], y[i])
        r_squared = r_value**2
        r2_scores_sum += r_squared
    
    # Calculate average R2 score
    avg_r2_score = r2_scores_sum / batch_size
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    return torch.tensor(1 - avg_r2_score, dtype = torch.float32).to(device)

def wasserstein_loss(output, target):
    """
    Calculate the Wasserstein loss between model output and target.
    
    The loss is defined as the mean of the element-wise product.
    """
    return torch.mean(output * target)

def mcc_loss(x, y):
    """
    Compute the Matthews Correlation Coefficient (MCC) loss between tensors x and y.
    
    The MCC is a correlation coefficient between the observed and predicted binary classifications.
    This implementation adapts it for continuous values by treating them as correlation measures.
    The loss is defined as the negative correlation coefficient to make it suitable for minimization.
    
    Parameters:
        x (torch.Tensor): Predicted values tensor
        y (torch.Tensor): Target values tensor
        
    Returns:
        torch.Tensor: Negative correlation coefficient as a loss value
    """
    x_mean = torch.mean(x)
    y_mean = torch.mean(y)
    # Center the inputs by subtracting their means
    x_centered = x - x_mean
    y_centered = y - y_mean
    # Calculate the numerator (covariance)
    numerator = torch.sum(x_centered * y_centered)
    # Add small epsilon to avoid division by zero
    epsilon = 1e-8
    # Calculate the denominator (product of standard deviations)
    denominator = torch.sqrt(torch.sum(x_centered ** 2)) * torch.sqrt(torch.sum(y_centered ** 2)) + epsilon
    # Calculate Pearson correlation coefficient
    correlation = numerator / denominator
    # Return negative correlation as loss value
    return -correlation

def cycle_loss(
    y_gene, x_gene, 
    y_perturbed_gene, x_perturbed_gene, 
    y_batch, x_batch,
    lambda_exp=1, lambda_ko=0.1, lambda_batch=0.1
):
    """
    Calculate the combined cycle consistency loss for gene expression, perturbation, and batch effects.
    
    This function computes a weighted sum of three different loss components to ensure cycle consistency
    in gene expression space, perturbation space, and batch identification.
    
    Parameters:
        y_gene (torch.Tensor): Target gene expression values
        x_gene (torch.Tensor): Predicted gene expression values
        y_perturbed_gene (torch.Tensor): Target perturbation indicators
        x_perturbed_gene (torch.Tensor): Predicted perturbation indicators
        y_batch (torch.Tensor): Target batch identifiers
        x_batch (torch.Tensor): Predicted batch identifiers
        lambda_exp (float, optional): Weight for gene expression loss. Defaults to 1.
        lambda_ko (float, optional): Weight for perturbation loss. Defaults to 0.1.
        lambda_batch (float, optional): Weight for batch identification loss. Defaults to 0.1.
        
    Returns:
        torch.Tensor: Weighted sum of the three loss components
    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # ========== 1) Gene expression loss (MSE example) ========== #
    # MSE with broadcast shape
    criterion_gene = nn.MSELoss().to(device)  
    gene_loss = criterion_gene(x_gene, y_gene)
    
    # ========== 2) Perturbed gene BCE ========== #
    # Binary cross-entropy for perturbation indicators
    criterion_perturbed = nn.BCEWithLogitsLoss()
    perturbed_gene_loss = criterion_perturbed(x_perturbed_gene, y_perturbed_gene)
    
    # ========== 3) batch BCE ========== #
    # Binary cross-entropy for batch identifiers
    criterion_batch = nn.BCEWithLogitsLoss()
    batch_loss = criterion_batch(x_batch, y_batch)
    
    # Weighted sum of all loss components
    total_loss = (lambda_exp * gene_loss
                  + lambda_ko * perturbed_gene_loss
                  + lambda_batch * batch_loss)
    return total_loss


def l1_loss(x, y):
    """
    Compute the L1 loss (mean absolute error) between two tensors.
    
    L1 loss measures the average absolute difference between elements in x and y,
    which is less sensitive to outliers compared to L2 loss.

    Parameters:
        x (torch.Tensor): Predicted values tensor
        y (torch.Tensor): Target values tensor

    Returns:
        torch.Tensor: The L1 loss value (mean absolute error)
    """
    return F.l1_loss(x, y)


def loss_fn(x, y):
    """
    Compute the mean squared error (MSE) between two tensors.
    
    MSE measures the average squared difference between elements in x and y,
    which is commonly used as a loss function for regression problems.

    Parameters:
        x (torch.Tensor): Predicted values tensor
        y (torch.Tensor): Target values tensor

    Returns:
        torch.Tensor: The MSE loss value
    """
    return torch.mean((x - y) ** 2)


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor, reduction: str = 'sum') -> torch.Tensor:
    """
    Compute KL divergence between q(z|x) = N(mu, sigma^2) and p(z)=N(0,I).
    
    This function calculates the Kullback-Leibler divergence between a learned
    distribution q(z|x) parameterized by mu and logvar, and a standard normal
    distribution p(z). It's commonly used in variational autoencoders (VAEs).

    Parameters:
        mu (torch.Tensor): Mean of the latent distribution with shape (batch, latent_dim)
        logvar (torch.Tensor): Log-variance of the latent distribution with shape (batch, latent_dim)
        reduction (str): Reduction method - 'sum', 'mean', or 'none'. Defaults to 'sum'

    Returns:
        torch.Tensor: KL divergence value as a scalar (if reduction is 'sum' or 'mean')
                     or vector (if reduction is 'none')
    """
    # KL = 0.5 * (exp(logvar) + mu^2 - 1 - logvar)
    kl = 0.5 * (torch.exp(logvar) + mu**2 - 1.0 - logvar)
    if reduction == 'sum':
        return kl.sum()
    elif reduction == 'mean':
        return kl.mean()
    else:
        return kl  # shape=(batch, latent_dim)