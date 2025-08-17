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
