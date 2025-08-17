import torch

def is_tensor(x):
    """Check if input is a PyTorch tensor."""
    return isinstance(x, torch.Tensor)

def has_nan(x):
    """Check if tensor contains NaN values."""
    return is_tensor(x) and torch.isnan(x).any().item()

def has_inf(x):
    """Check if tensor contains Inf or -Inf values."""
    return is_tensor(x) and torch.isinf(x).any().item()

def check_tensor_validity(x, raise_error=False):
    """
    Check tensor validity: is tensor, no NaN, no Inf.
    Raises errors if raise_error=True.
    Returns status message otherwise.
    """
    if not is_tensor(x):
        msg = "Input is not a PyTorch Tensor."
        if raise_error:
            raise TypeError(msg)
        return msg

    if has_nan(x):
        msg = "NaN found in tensor!"
        if raise_error:
            raise ValueError(msg)
        return msg

    if has_inf(x):
        msg = "Inf found in tensor!"
        if raise_error:
            raise ValueError(msg)
        return msg

    return "Valid Tensor"
