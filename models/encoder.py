# encoder.py
import torch
from torch import nn
from .residual import ResidualBlock
from .attention import SelfAttention
from .utils import check_tensor


class _Encoder(nn.Module):
    """
    Encoder network that maps gene expression data and condition information
    to multiple latent spaces (expression, knockout, noise).
    """
    def __init__(
        self,
        gene_size,
        num_perturbed_gene,
        num_batch,
        embedding_dim=32,
        latent_dim=128,
        dropout_rate=0.1,
        num_res_blocks=2,
        use_attention=True,
        use_embedding=False,
        use_batch_norm=False,
        use_dropout=False
    ):
        """
        Initialize the encoder network.
        
        Args:
            gene_size (int): Number of genes in the expression data
            num_perturbed_gene (int): Number of possible perturbed genes
            num_batch (int): Number of possible batch conditions
            embedding_dim (int): Dimension for condition embeddings
            latent_dim (int): Dimension of the latent spaces
            dropout_rate (float): Dropout probability
            num_res_blocks (int): Number of residual blocks to use
            use_attention (bool): Whether to use self-attention
            use_embedding (bool): Whether to use embedding or one-hot for conditions
            use_batch_norm (bool): Whether to use batch normalization
            use_dropout (bool): Whether to use dropout
        """
        super().__init__()
        self.gene_size = gene_size
        self.embedding_dim = embedding_dim
        self.latent_dim = latent_dim
        self.use_attention = use_attention
        self.use_embedding = use_embedding
        self.use_batch_norm = use_batch_norm
        self.use_dropout = use_dropout
        # Condition embedding
        if use_embedding:
            self.perturbed_gene_embedding = nn.Embedding(num_perturbed_gene, embedding_dim)
            self.batch_embedding = nn.Embedding(num_batch, embedding_dim)
        else:
            self.perturbed_gene_embedding = nn.Linear(num_perturbed_gene, embedding_dim)
            self.batch_embedding = nn.Linear(num_batch, embedding_dim)

        # Input: gene expression + perturbed gene embedding + batch embedding
        input_dim = gene_size + 2 * embedding_dim

        self.fc1 = nn.Linear(input_dim, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.prelu1 = nn.PReLU()
        self.drop1 = nn.Dropout(dropout_rate)

        self.fc2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.prelu2 = nn.PReLU()
        self.drop2 = nn.Dropout(dropout_rate)

        self.fc3 = nn.Linear(512, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.prelu3 = nn.PReLU()
        self.drop3 = nn.Dropout(dropout_rate)

        # Multiple ResidualBlocks
        self.res_blocks = nn.ModuleList([
            ResidualBlock(256, dropout_rate=dropout_rate) for _ in range(num_res_blocks)
        ])

        # Optional SelfAttention
        self.attention = SelfAttention(256) if use_attention else None

        # Output 6 vectors (mu/logvar) for expression, knockout, noise
        self.fc_mu_exp       = nn.Linear(256, latent_dim)
        self.fc_logvar_exp   = nn.Linear(256, latent_dim)
        self.fc_mu_ko        = nn.Linear(256, latent_dim)
        self.fc_logvar_ko    = nn.Linear(256, latent_dim)
        self.fc_mu_noise     = nn.Linear(256, latent_dim)
        self.fc_logvar_noise = nn.Linear(256, latent_dim)

    def forward(self, x, perturbed_gene_idx, batch_idx):
        """
        Forward pass through the encoder.
        
        Args:
            x (torch.Tensor): Gene expression data
            perturbed_gene_idx (torch.Tensor): Indices of perturbed genes
            batch_idx (torch.Tensor): Indices of batch conditions
            
        Returns:
            tuple: Six tensors representing means and log variances for the three latent spaces:
                  (mu_exp, logvar_exp, mu_ko, logvar_ko, mu_noise, logvar_noise)
        """
        check_tensor(x, 'x')
        e_p = self.perturbed_gene_embedding(perturbed_gene_idx)
        e_b = self.batch_embedding(batch_idx)

        # Concatenate inputs
        h = torch.cat([x, e_p, e_b], dim=1)

        # 3 FC layers
        h = self.fc1(h)
        if self.use_batch_norm:
            h = self.bn1(h)
        h = self.prelu1(h)
        if self.use_dropout:
            h = self.drop1(h)

        h = self.fc2(h)
        if self.use_batch_norm:
            h = self.bn2(h)
        h = self.prelu2(h)
        if self.use_dropout:
            h = self.drop2(h)

        h = self.fc3(h)
        if self.use_batch_norm:
            h = self.bn3(h)
        h = self.prelu3(h)
        if self.use_dropout:
            h = self.drop3(h)

        # Multiple residual blocks
        for block in self.res_blocks:
            h = block(h)

        # Optional SelfAttention
        if self.use_attention and self.attention is not None:
            h = self.attention(h)

        # Output 6 vectors
        mu_exp       = self.fc_mu_exp(h)
        logvar_exp   = self.fc_logvar_exp(h)
        mu_ko        = self.fc_mu_ko(h)
        logvar_ko    = self.fc_logvar_ko(h)
        mu_noise     = self.fc_mu_noise(h)
        logvar_noise = self.fc_logvar_noise(h)

        return (mu_exp, logvar_exp, mu_ko, logvar_ko, mu_noise, logvar_noise)

