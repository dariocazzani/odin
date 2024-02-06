import numpy as np
import pytest
import torch

from odin.inference_engines.ops import (
    elu,
    elu_torch,
    identity,
    identity_torch,
    relu,
    relu_torch,
    sigmoid,
    sigmoid_torch,
    tanh,
    tanh_torch,
)
from odin.interfaces.custom_types import (
    Float32,
    float32
)

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
