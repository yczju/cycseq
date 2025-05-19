# generator.py
import torch
from torch import nn
from .encoder import _Encoder
from .decoder import _Decoder
from .utils import reparameterize

class Generator(nn.Module):
    """
    Complete generator model combining encoder and decoder.
    Encodes input data to latent spaces and decodes back to reconstructions.
    """
    def __init__(
        self,
        gene_size,
        num_perturbed_gene,
        num_batch,
        embedding_dim=32,
        latent_dim=128,
        dropout_rate=0.2,
        num_res_blocks=2,
        use_attention=True,
        use_embedding=False,
        use_batch_norm=False, 
        use_dropout=False
    ):
        """
        Initialize the generator.
        
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
        self.encoder = _Encoder(
            gene_size, num_perturbed_gene, num_batch,
            embedding_dim, latent_dim, dropout_rate,
            num_res_blocks=num_res_blocks,
            use_attention=use_attention,
            use_embedding=use_embedding,
            use_batch_norm=use_batch_norm,
            use_dropout=use_dropout
        )
        self.decoder = _Decoder(
            gene_size, num_perturbed_gene, num_batch,
            embedding_dim, latent_dim, dropout_rate,
            num_res_blocks=num_res_blocks,
            use_attention=use_attention,
            use_embedding=use_embedding,
            use_batch_norm=use_batch_norm,
            use_dropout=use_dropout
        )
        self.use_embedding = use_embedding
        self.use_batch_norm = use_batch_norm
        self.use_dropout = use_dropout

    def forward(self, x, perturbed_gene_onehot, batch_onehot):
        """
        Forward pass through the generator.
        
        Args:
            x (torch.Tensor): Gene expression data
            perturbed_gene_onehot (torch.Tensor): One-hot or index tensor for perturbed genes
            batch_onehot (torch.Tensor): One-hot or index tensor for batch conditions
            
        Returns:
            tuple: Six elements:
                  x_hat: Reconstructed gene expression
                  p_hat: Predicted perturbed gene logits
                  b_hat: Predicted batch logits
                  (mu_exp, logvar_exp, mu_ko, logvar_ko, mu_noise, logvar_noise): Latent parameters
        """
        if self.use_embedding:
            perturbed_gene_idx = perturbed_gene_onehot.argmax(dim=1)
            batch_idx = batch_onehot.argmax(dim=1)
        else:
            perturbed_gene_idx = perturbed_gene_onehot
            batch_idx = batch_onehot

        (mu_exp, logvar_exp,
         mu_ko, logvar_ko,
         mu_noise, logvar_noise) = self.encoder(x, perturbed_gene_idx, batch_idx)

        z_exp   = reparameterize(mu_exp,   logvar_exp)
        z_ko    = reparameterize(mu_ko,    logvar_ko)
        z_noise = reparameterize(mu_noise, logvar_noise)

        x_hat, p_hat, b_hat = self.decoder(z_exp, z_ko, z_noise, perturbed_gene_idx, batch_idx)

        # Return reconstructions and latent parameters for KL divergence calculation
        return (x_hat, p_hat, b_hat,
                (mu_exp, logvar_exp, mu_ko, logvar_ko, mu_noise, logvar_noise))

