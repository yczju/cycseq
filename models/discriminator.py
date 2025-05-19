# discriminator.py
import torch
from torch import nn

class Discriminator(nn.Module):
    """
    Discriminator network for adversarial training.
    Distinguishes between real and generated data.
    """
    def __init__(
        self,
        gene_size,
        num_perturbed_gene,
        num_batch,
        hidden_dims=[1024, 512, 256],
        dropout=0.2
    ):
        """
        Initialize the discriminator.
        
        Args:
            gene_size (int): Number of genes in the expression data
            num_perturbed_gene (int): Number of possible perturbed genes
            num_batch (int): Number of possible batch conditions
            hidden_dims (list): Dimensions of hidden layers
            dropout (float): Dropout probability
        """
        super().__init__()
        # Input dimension = gene expression + perturbed gene (one-hot) + batch (one-hot)
        input_dim = gene_size + num_perturbed_gene + num_batch
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.prelu1 = nn.PReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.prelu2 = nn.PReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.fc3 = nn.Linear(hidden_dims[1], hidden_dims[2])
        self.prelu3 = nn.PReLU()
        self.dropout3 = nn.Dropout(dropout)

        self.fc4 = nn.Linear(hidden_dims[2], 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, perturbed_gene_onehot, batch_onehot):
        """
        Forward pass through the discriminator.
        
        Args:
            x (torch.Tensor): Gene expression data
            perturbed_gene_onehot (torch.Tensor): One-hot encoding of perturbed genes
            batch_onehot (torch.Tensor): One-hot encoding of batch conditions
            
        Returns:
            torch.Tensor: Probability that the input is real (1) vs. fake (0)
        """
        # Concatenate inputs
        h = torch.cat([x, perturbed_gene_onehot, batch_onehot], dim=1)

        h = self.prelu1(self.fc1(h))
        h = self.dropout1(h)

        h = self.prelu2(self.fc2(h))
        h = self.dropout2(h)

        h = self.prelu3(self.fc3(h))
        h = self.dropout3(h)

        out = self.sigmoid(self.fc4(h))  # For adversarial discrimination, sigmoid outputs [0,1]
        return out

