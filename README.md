# tensor-toolkit
![PyPI](https://img.shields.io/pypi/v/tensor-toolkit)
![Python](https://img.shields.io/pypi/pyversions/tensor-toolkit)
![License](https://img.shields.io/badge/license-MIT-green)

A lightweight PyTorch utility toolkit for tensor operations, validation, and model utilities.

---

## Features and Function Descriptions

### Tensor Utilities

- **`to_device(tensor, device=None)`**  
  Move a tensor to the specified device (CPU or GPU). Automatically selects GPU if available when no device is provided.

- **`get_device(tensor)`**  
  Returns the current device of the given tensor.

- **`seed_all(seed=42)`**  
  Set random seeds for PyTorch (CPU and GPU) and NumPy to ensure reproducible results.

- **`count_params(model, trainable_only=True)`**  
  Count the number of parameters in a PyTorch model. Optionally count only trainable parameters.

- **`tensor_summary(tensor)`**  
  Print a detailed summary of a tensor including shape, dtype, device, and statistics (min, max, mean, std).

- **`one_hot(labels, num_classes)`**  
  Convert a tensor of label indices into one-hot encoded format.

- **`accuracy(outputs, targets)`**  
  Compute classification accuracy by comparing model outputs (logits or probabilities) to target labels.

---

### Tensor Checks 

- **`is_tensor(x)`**  
  Check whether an object is a PyTorch tensor.

- **`has_nan(x)`**  
  Check if the tensor contains any NaN values.

- **`has_inf(x)`**  
  Check if the tensor contains any positive or negative infinity values.

- **`check_tensor_validity(x, raise_error=False)`**  
  Validate the tensor for correct type, and absence of NaN or Inf values. Raises exceptions if `raise_error=True`; otherwise, returns status messages.

---

## Installation

```bash
pip install tensor-toolkit
```
## Requirements

- torch >=1.10.0
- numpy >=1.21.0 (used for seeding and numerical utilities)

To install requirements:

```bash
pip install -r requirements.txt
```

---

## Usage Example
```python
import torch
import tensorkit as tk

def main():
    # Create a random tensor
    x = torch.randn(3, 3)
    print("Original tensor device:", tk.get_device(x))

    # Move tensor to device (GPU if available)
    x = tk.to_device(x)
    print("Tensor moved to device:", tk.get_device(x))

    # Set seed for reproducibility
    tk.seed_all(42)
    t1 = torch.randn(2, 2)
    tk.seed_all(42)
    t2 = torch.randn(2, 2)
    print("Seed reproducible two tensors equal:", torch.equal(t1, t2))

    # Count parameters of a model
    model = torch.nn.Linear(4, 2)
    print(f"Model trainable parameters: {tk.count_params(model)}")

    # Print tensor summary
    print("Tensor summary:")
    tk.tensor_summary(x)

    # One-hot encode labels
    labels = torch.tensor([0, 1, 2])
    one_hot_labels = tk.one_hot(labels, num_classes=3)
    print("One-hot encoded labels:\n", one_hot_labels)

    # Calculate accuracy given model outputs and targets
    outputs = torch.tensor([[0.8, 0.1, 0.1],
                            [0.2, 0.7, 0.1],
                            [0.1, 0.2, 0.7]])
    targets = torch.tensor([0, 1, 2])
    acc = tk.accuracy(outputs, targets)
    print(f"Accuracy: {acc:.2%}")

    # Check tensor validity
    print("Is tensor:", tk.is_tensor(x))
    print("Contains NaN:", tk.has_nan(x))
    print("Contains Inf:", tk.has_inf(x))
    print("Tensor validity:", tk.check_tensor_validity(x))

if __name__ == "__main__":
    main()
```
---

## Project Structure
```
tensor-toolkit/             # Project root folder
├── examples/               # Example scripts
│ └── usage_example.py      # Usage demonstration
├── tensorkit/              # Main package folder
│ ├── init.py               # API exports
│ ├── tensor_utils.py       # Tensor utility functions
│ ├── tensor_checks.py      # Tensor validation functions
├── tests/                  # Unit tests
│ ├── init.py
│ ├── test_tensor_utils.py  # Tests for tensor_utils
│ └── test_tensor_checks.py # Tests for tensor_checks
├── LICENSE                 # MIT License
├── pyproject.toml          # Project configuration
├── README.md               # Project documentation
├── requirements.txt        # Package dependencies
```
---

## Contact

Created by Suresh K — [GitHub](https://github.com/suressssz/tensor-toolkit)

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
