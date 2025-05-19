# scheduler.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch import optim



def unfreeze_layers(epoch, model, unfreeze_every=10):
    """
    Gradually unfreeze layers in the encoder of the specified model every 'unfreeze_every' epochs.
    
    Parameters:
        epoch (int): The current epoch number.
        model (nn.Module): The neural network model containing an encoder with Linear layers.
        unfreeze_every (int): Frequency (in epochs) to unfreeze an additional layer.

    The function unfreezes layers in reverse order (from the last to the first) and avoids
    duplicate unfreezing.
    """
    if epoch % unfreeze_every == 0 and epoch != 0:
        # Determine the target number of layers to unfreeze based on the current epoch.
        layers_to_unfreeze = epoch // unfreeze_every
        # Count the currently unfrozen linear layers in the encoder.
        current_unfrozen_layers = sum(
            1 for layer in model.encoder.modules() 
            if isinstance(layer, nn.Linear) and any(p.requires_grad for p in layer.parameters())
        )
        # Unfreeze additional layers if necessary.
        if current_unfrozen_layers < layers_to_unfreeze:
            linear_layers = [layer for layer in model.encoder.modules() if isinstance(layer, nn.Linear)]
            linear_layers.reverse()  # Unfreeze from the last layer first.
            try:
                layer_to_unfreeze = linear_layers[current_unfrozen_layers]
            except IndexError:
                raise IndexError("Attempting to unfreeze more layers than available in the encoder.")
            for param in layer_to_unfreeze.parameters():
                param.requires_grad = True
            print(f"Epoch {epoch}: Unfroze layer {current_unfrozen_layers + 1} parameters")
