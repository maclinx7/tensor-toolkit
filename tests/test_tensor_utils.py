import torch
import numpy as np
import pytest

from tensorkit.tensor_utils import (
    to_device, get_device, seed_all, count_params,
    tensor_summary, one_hot, accuracy
)


def test_to_device_and_get_device():
    x = torch.randn(2, 2)
    x_cpu = to_device(x, "cpu")
    assert get_device(x_cpu) == torch.device("cpu")
    if torch.cuda.is_available():
        x_cuda = to_device(x, "cuda")
        assert get_device(x_cuda) == torch.device("cuda")


def test_seed_all_reproducibility():
    seed_all(42)
    t1 = torch.randn(3, 3)
    np1 = np.random.rand(3)
    seed_all(42)
    t2 = torch.randn(3, 3)
    np2 = np.random.rand(3)
    assert torch.equal(t1, t2)
    assert np.allclose(np1, np2)


def test_count_params():
    model = torch.nn.Linear(10, 5)
    trainable_count = count_params(model, trainable_only=True)
    total_count = count_params(model, trainable_only=False)
    manual_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    manual_total = sum(p.numel() for p in model.parameters())
    assert trainable_count == manual_trainable
    assert total_count == manual_total


def test_tensor_summary(capsys):
    x = torch.randn(5, 5)
    tensor_summary(x)
    output = capsys.readouterr().out
    assert "Shape" in output and "Dtype" in output
    assert "Device" in output and "Min" in output and "Max" in output
    assert "Mean" in output and "Std" in output

    empty = torch.tensor([])
    tensor_summary(empty)
    empty_output = capsys.readouterr().out
    assert "Tensor is empty." in empty_output


def test_one_hot():
    labels = torch.tensor([0, 2, 1])
    oh = one_hot(labels, num_classes=3)
    assert oh.shape == (3, 3)
    expected = torch.tensor([[1, 0, 0],
                             [0, 0, 1],
                             [0, 1, 0]], dtype=oh.dtype)
    assert torch.equal(oh, expected)


def test_accuracy():
    outputs = torch.tensor([[0.9, 0.05, 0.05],
                            [0.1, 0.8, 0.1],
                            [0.05, 0.05, 0.9]])
    targets = torch.tensor([0, 1, 2])
    acc = accuracy(outputs, targets)
    assert acc == pytest.approx(1.0)

    # Test with some wrong predictions (2 out of 3 correct)
    outputs_wrong = torch.tensor([[0.9, 0.5, 0.5],
                                  [0.2, 0.7, 0.1],
                                  [0.05, 0.9, 0.05]])
    acc_wrong = accuracy(outputs_wrong, targets)
    assert acc_wrong == pytest.approx(2/3)
