import torch
import numpy as np

def to_device(tensor, device=None):
    """Move tensor to given device (CPU/GPU). If None, auto-detect."""
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return tensor.to(device)

def get_device(tensor):
    """Return device of given tensor as torch.device."""
    return tensor.device

def seed_all(seed=42):
    """Set random seed for reproducibility (CPU + GPU + NumPy)."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def count_params(model, trainable_only=True):
    """Count number of parameters in a model."""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())

def tensor_summary(tensor):
    """Print summary of a tensor: shape, dtype, device, stats (min, max, mean, std)."""
    if tensor.numel() == 0:
        print("Tensor is empty.")
        return

    print(f"Shape: {tuple(tensor.shape)}")
    print(f"Dtype: {tensor.dtype}")
    print(f"Device: {tensor.device}")
    print(f"Min: {tensor.min().item():.4f}, Max: {tensor.max().item():.4f}")
    print(f"Mean: {tensor.mean().item():.4f}, Std: {tensor.std().item():.4f}")

def one_hot(labels, num_classes):
    """Convert label indices to one-hot encoding."""
    return torch.nn.functional.one_hot(labels, num_classes=num_classes)

def accuracy(outputs, targets):
    """Compute accuracy for classification from raw outputs and labels."""
    preds = torch.argmax(outputs, dim=1)
    return (preds == targets).float().mean().item()
