# decoder.py
import torch
from torch import nn
from .residual import ResidualBlock
from .attention import SelfAttention
from .utils import check_tensor

class _Decoder(nn.Module):
    """
    Decoder network that maps latent representations and condition information
    back to gene expression data and condition predictions.
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
        Initialize the decoder network.
        
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

        # Input: 3 latent vectors + 2 condition embeddings
        input_dim = 3 * latent_dim + 2 * embedding_dim

        self.tanh = nn.Tanh()

        self.fc1 = nn.Linear(input_dim, 1024)
        if use_batch_norm:
            self.bn1 = nn.BatchNorm1d(1024)
        self.prelu1 = nn.PReLU()
        self.drop1 = nn.Dropout(dropout_rate)

        self.fc2 = nn.Linear(1024, 512)
        if use_batch_norm:
            self.bn2 = nn.BatchNorm1d(512)
        self.prelu2 = nn.PReLU()
        self.drop2 = nn.Dropout(dropout_rate)

        self.fc3 = nn.Linear(512, 256)
        if use_batch_norm:
            self.bn3 = nn.BatchNorm1d(256)
        self.prelu3 = nn.PReLU()
        self.drop3 = nn.Dropout(dropout_rate)

        # Multiple ResidualBlocks
        self.res_blocks = nn.ModuleList([
            ResidualBlock(256, dropout_rate=dropout_rate) for _ in range(num_res_blocks)
        ])

        # Optional SelfAttention
        self.attention = SelfAttention(256) if use_attention else None

        # Output three parts
        self.fc_out_exp = nn.Linear(256, gene_size)
        if use_batch_norm:
            self.fc_out_perturbed = nn.Linear(256, num_perturbed_gene)
            self.fc_out_batch     = nn.Linear(256, num_batch)
        else:
            self.fc_out_perturbed = nn.Sequential(nn.Linear(256, num_perturbed_gene), nn.Tanh())
            self.fc_out_batch     = nn.Sequential(nn.Linear(256, num_batch), nn.Tanh())

    def forward(self, z_exp, z_ko, z_noise, perturbed_gene_idx, batch_idx):
        """
        Forward pass through the decoder.
        
        Args:
            z_exp (torch.Tensor): Latent vector for expression
            z_ko (torch.Tensor): Latent vector for knockout
            z_noise (torch.Tensor): Latent vector for noise
            perturbed_gene_idx (torch.Tensor): Indices of perturbed genes
            batch_idx (torch.Tensor): Indices of batch conditions
            
        Returns:
            tuple: Three tensors:
                  x_hat: Reconstructed gene expression
                  p_hat: Predicted perturbed gene logits
                  b_hat: Predicted batch logits
        """
        check_tensor(z_exp, 'z_exp')
        check_tensor(z_ko, 'z_ko')
        check_tensor(z_noise, 'z_noise')

        e_p = self.perturbed_gene_embedding(perturbed_gene_idx)
        e_b = self.batch_embedding(batch_idx)

        # Concatenate
        h = torch.cat([z_exp, z_ko, z_noise, e_p, e_b], dim=1)

        # 3 FC layers
        h = self.fc1(h)
        # h = self.bn1(h)
        h = self.prelu1(h)
        if self.use_dropout:
            h = self.drop1(h)

        h = self.fc2(h)
        # h = self.bn2(h)
        h = self.prelu2(h)
        if self.use_dropout:
            h = self.drop2(h)

        h = self.fc3(h)
        # h = self.bn3(h)
        h = self.prelu3(h)
        if self.use_dropout:
            h = self.drop3(h)

        # Multiple residual blocks
        for block in self.res_blocks:
            h = block(h)

        # Optional SelfAttention
        if self.use_attention and self.attention is not None:
            h = self.attention(h)

        # Split output into three parts
        # Apply tanh to gene expression as per paper
        x_hat = self.tanh(self.fc_out_exp(h))  # gene expression

        p_hat = self.fc_out_perturbed(h)     # perturbed gene logits
        b_hat = self.fc_out_batch(h)         # batch logits

        return x_hat, p_hat, b_hat

