# hooks.py
import torch

def register_debug_hooks(model):
    """
    Register forward hooks on all non-container layers of the model to monitor
    the intermediate outputs. This aids in debugging by checking for NaN or infinite
    values as well as logging basic statistics (mean and standard deviation).

    Parameters:
        model (nn.Module): The neural network model.

    Returns:
        list: A list of registered hook handles.
    """
    hooks = []
    
    def forward_hook(module, input, output, name):
        # Check for NaN or Inf values in the output tensor.
        has_nan = torch.isnan(output).any().item()
        has_inf = torch.isinf(output).any().item()
        output_val = output[0] if isinstance(output, tuple) else output
        mean_value = output_val.mean().item()
        std_value = output_val.std().item()
        
        print(f"[Layer {name}] Mean: {mean_value:.4f}, Std: {std_value:.4f}, NaN: {has_nan}, Inf: {has_inf}")
        if has_nan or has_inf:
            print(f"!!! NaN/Inf detected in layer {name} !!!")
            import pdb; pdb.set_trace()
    
    # Register the forward hook for each non-container module.
    for name, module in model.named_modules():
        if list(module.children()):
            continue
        hook = module.register_forward_hook(lambda m, i, o, name=name: forward_hook(m, i, o, name))
        hooks.append(hook)
    
    return hooks