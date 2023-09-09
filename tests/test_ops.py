import pytest
import torch
import numpy as np
from inference_engines.ops import sigmoid_torch, relu_torch, tanh_torch, identity_torch, elu_torch
from inference_engines.ops import sigmoid, relu, tanh, identity, elu
from interfaces.custom_types import float32, Float32

np.random.seed(53)
@pytest.mark.parametrize("activation_torch, activation_numpy", [
    (sigmoid_torch, sigmoid),
    (relu_torch, relu),
    (tanh_torch, tanh),
    (identity_torch, identity),
    (elu_torch, elu)
])
def test_activation_functions(activation_torch, activation_numpy):
    for multiplier in range(-10, 10):
        np_input:Float32 = float32(np.random.randn()) * multiplier
        torch_input = torch.Tensor([np_input])
        np_result = activation_numpy(np_input)
        torch_result = activation_torch(torch_input).cpu().detach().numpy()
        assert np.allclose(np_result, torch_result, atol=1e-7)
