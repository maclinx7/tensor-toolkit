import torch
import pytest

from tensorkit.tensor_checks import (
    is_tensor,
    has_nan,
    has_inf,
    check_tensor_validity
)

def test_is_tensor():
    assert is_tensor(torch.tensor([1]))
    assert not is_tensor([1, 2, 3])
    assert not is_tensor(123)
    assert not is_tensor(None)

def test_has_nan():
    x = torch.tensor([1.0, float('nan')])
    y = torch.tensor([1.0, 2.0])
    assert has_nan(x) is True
    assert has_nan(y) is False
    # Non-tensor inputs always False
    assert has_nan([float('nan')]) is False

def test_has_inf():
    x = torch.tensor([1.0, float('inf')])
    y = torch.tensor([1.0, -float('inf')])
    z = torch.tensor([1.0, 2.0])
    assert has_inf(x) is True
    assert has_inf(y) is True
    assert has_inf(z) is False
    # Non-tensor inputs always False
    assert has_inf([float('inf')]) is False

def test_check_tensor_validity_returns():
    valid_tensor = torch.tensor([1.0, 2.0])
    tensor_with_nan = torch.tensor([1.0, float('nan')])
    tensor_with_inf = torch.tensor([float('inf'), 2.0])
    not_tensor = [1.0, 2.0]

    assert check_tensor_validity(valid_tensor) == "Valid Tensor"
    assert check_tensor_validity(tensor_with_nan) == "NaN found in tensor!"
    assert check_tensor_validity(tensor_with_inf) == "Inf found in tensor!"
    assert check_tensor_validity(not_tensor) == "Input is not a PyTorch Tensor."

def test_check_tensor_validity_raises():
    valid_tensor = torch.tensor([1.0, 2.0])
    tensor_with_nan = torch.tensor([1.0, float('nan')])
    tensor_with_inf = torch.tensor([float('inf'), 2.0])
    not_tensor = [1.0, 2.0]

    # No error for valid tensor
    assert check_tensor_validity(valid_tensor, raise_error=True) == "Valid Tensor"

    with pytest.raises(TypeError, match="Input is not a PyTorch Tensor."):
        check_tensor_validity(not_tensor, raise_error=True)

    with pytest.raises(ValueError, match="NaN found in tensor!"):
        check_tensor_validity(tensor_with_nan, raise_error=True)

    with pytest.raises(ValueError, match="Inf found in tensor!"):
        check_tensor_validity(tensor_with_inf, raise_error=True)
