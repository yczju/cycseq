# models/checkpoint.py
import os
import itertools
import torch
from torch.optim.lr_scheduler import StepLR
from torch import optim

from .generator import Generator
from .discriminator import Discriminator
from ..data.harmony import get_full_gene_list, get_batch_label_list

def save_checkpoint(epoch, G, F_net, D_X, D_Y, optimizer_G, optimizer_D_X, optimizer_D_Y, 
                    scheduler_G, scheduler_D_X, scheduler_D_Y, best_val_loss, trigger_times, filename):
    """
    Save the current training checkpoint including model states, optimizer states, 
    scheduler states, and training metrics.

    Parameters:
        epoch (int): The current training epoch.
        G, F_net, D_X, D_Y (nn.Module): Neural network models.
        optimizer_G, optimizer_D_X, optimizer_D_Y (torch.optim.Optimizer): Optimizers for the models.
        scheduler_G, scheduler_D_X, scheduler_D_Y: Learning rate schedulers.
        best_val_loss (float): The best validation loss recorded so far.
        trigger_times (int): The number of times early stopping has been triggered.
        filename (str): The file path where the checkpoint will be saved.
    """
    # Construct the checkpoint dictionary.
    state = {
        'epoch': epoch,
        'G_state_dict': G.state_dict(),
        'F_state_dict': F_net.state_dict(),
        'D_X_state_dict': D_X.state_dict(),
        'D_Y_state_dict': D_Y.state_dict(),
        'optimizer_G_state_dict': optimizer_G.state_dict(),
        'optimizer_D_X_state_dict': optimizer_D_X.state_dict(),
        'optimizer_D_Y_state_dict': optimizer_D_Y.state_dict(),
        'scheduler_G_state_dict': scheduler_G.state_dict(),
        'scheduler_D_X_state_dict': scheduler_D_X.state_dict(),
        'scheduler_D_Y_state_dict': scheduler_D_Y.state_dict(),
        'best_val_loss': best_val_loss,
        'trigger_times': trigger_times
    }

    # Ensure the directory for saving the checkpoint exists.
    checkpoint_dir = os.path.dirname(filename)
    if checkpoint_dir and not os.path.exists(checkpoint_dir):
        try:
            os.makedirs(checkpoint_dir, exist_ok=True)
        except Exception as e:
            raise IOError(f"Failed to create directory '{checkpoint_dir}': {e}")

    # Save the checkpoint to the specified file.
    try:
        torch.save(state, filename)
        print(f'Checkpoint successfully saved to {filename}')
    except Exception as e:
        raise IOError(f"Failed to save checkpoint to '{filename}': {e}")


def load_checkpoint(num_res_blocks, use_attention, use_embedding, use_batch_norm, use_dropout, filename):
    """
    Load a training checkpoint and restore the neural network models, optimizers,
    and schedulers.

    Parameters:
        G, F_net, D_X, D_Y (nn.Module): Neural network models.
        optimizer_G, optimizer_D_X, optimizer_D_Y (torch.optim.Optimizer): Optimizers.
        scheduler_G, scheduler_D_X, scheduler_D_Y: Learning rate schedulers.
        filename (str): The file path from which to load the checkpoint.

    Returns:
        tuple: start_epoch, updated models, optimizers, schedulers,
               best_val_loss, and trigger_times.
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Checkpoint file not found: {filename}")
    try:
        checkpoint = torch.load(filename)
    except Exception as e:
        raise IOError(f"Failed to load checkpoint from '{filename}': {e}")

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    full_gene_list = get_full_gene_list()
    batch_label_list = get_batch_label_list()
    gene_size = len(full_gene_list)
    num_perturbed_gene = len(full_gene_list)
    num_batch = len(batch_label_list)

    G = Generator(
        gene_size=gene_size, 
        num_perturbed_gene=num_perturbed_gene, 
        num_batch=num_batch,
        num_res_blocks=num_res_blocks,
        use_attention=use_attention,
        use_embedding=use_embedding,
        use_batch_norm=use_batch_norm,
        use_dropout=use_dropout
    ).to(device)
    F = Generator(
        gene_size=gene_size, 
        num_perturbed_gene=num_perturbed_gene, 
        num_batch=num_batch,
        num_res_blocks=num_res_blocks,
        use_attention=use_attention,
        use_embedding=use_embedding,
        use_batch_norm=use_batch_norm,
        use_dropout=use_dropout
    ).to(device)    
    D_X = Discriminator(gene_size=gene_size, num_perturbed_gene=num_perturbed_gene, num_batch=num_batch).to(device)
    D_Y = Discriminator(gene_size=gene_size, num_perturbed_gene=num_perturbed_gene, num_batch=num_batch).to(device)
    optimizer_G = optim.Adam(itertools.chain(G.parameters(), F.parameters()), lr=0.0002, betas=(0.5, 0.9))
    optimizer_D_X = optim.Adam(D_X.parameters(), lr=0.0002, betas=(0.5, 0.9))
    optimizer_D_Y = optim.Adam(D_Y.parameters(), lr=0.0002, betas=(0.5, 0.9))
    scheduler_G = StepLR(optimizer_G, step_size=10, gamma=0.5)
    scheduler_D_X = StepLR(optimizer_D_X, step_size=10, gamma=0.5)
    scheduler_D_Y = StepLR(optimizer_D_Y, step_size=10, gamma=0.5)

    # Restore model states.
    G.load_state_dict(checkpoint.get('G_state_dict', {}))
    F.load_state_dict(checkpoint.get('F_state_dict', {}))
    D_X.load_state_dict(checkpoint.get('D_X_state_dict', {}))
    D_Y.load_state_dict(checkpoint.get('D_Y_state_dict', {}))

    # Restore optimizer states.
    optimizer_G.load_state_dict(checkpoint.get('optimizer_G_state_dict', {}))
    optimizer_D_X.load_state_dict(checkpoint.get('optimizer_D_X_state_dict', {}))
    optimizer_D_Y.load_state_dict(checkpoint.get('optimizer_D_Y_state_dict', {}))

    # Restore scheduler states.
    scheduler_G.load_state_dict(checkpoint.get('scheduler_G_state_dict', {}))
    scheduler_D_X.load_state_dict(checkpoint.get('scheduler_D_X_state_dict', {}))
    scheduler_D_Y.load_state_dict(checkpoint.get('scheduler_D_Y_state_dict', {}))

    best_val_loss = checkpoint.get('best_val_loss', None)
    trigger_times = checkpoint.get('trigger_times', None)
    epoch = checkpoint.get('epoch', None)
    return G, F, (epoch, D_X, D_Y, optimizer_G, optimizer_D_X, optimizer_D_Y, \
           scheduler_G, scheduler_D_X, scheduler_D_Y, best_val_loss, trigger_times)
