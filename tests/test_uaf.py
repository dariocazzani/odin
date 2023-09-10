import numpy as np
import torch
from inference_engines.ops import UAF
from inference_engines.ops import UAF_Torch
from interfaces.custom_types import float32, Float32

def test_relu():
    uaf = UAF()
    assert uaf.relu(float32(1.0)) == float32(1.0)
    assert uaf.relu(float32(-1.0)) == float32(0.0)
    assert uaf.relu(float32(0.0)) == float32(0.0)

def test_perturbe_parameter():
    uaf = UAF()
    initial_A = uaf.A
    perturbation = float32(0.1)
    uaf.perturbe_parameter("A", perturbation)
    assert uaf.A == initial_A + perturbation

def test_call():
    uaf = UAF()
    input_val = float32(1.0)
    expected_output = uaf.relu(uaf.A * (input_val + uaf.B) + uaf.C * np.square(input_val)) \
                      + np.log1p(np.exp(-np.abs(uaf.A * (input_val + uaf.B) + uaf.C * np.square(input_val)))) \
                      - uaf.relu(uaf.D * (input_val - uaf.B)) \
                      - np.log1p(np.exp(-np.abs(uaf.D * (input_val - uaf.B)))) \
                      + uaf.E
    assert uaf(input_val) == expected_output
    
def test_consistency():
    uaf = UAF()
    uaf_torch = UAF_Torch()

    # Making sure that parameters in both models are the same
    uaf_torch.A.data.fill_(float(uaf.A))
    uaf_torch.B.data.fill_(float(uaf.B))
    uaf_torch.C.data.fill_(float(uaf.C))
    uaf_torch.D.data.fill_(float(uaf.D))
    uaf_torch.E.data.fill_(float(uaf.E))

    single_input_val = float32(1.0)
    single_input_torch = torch.tensor([single_input_val], dtype=torch.float32)
    assert np.isclose(uaf(single_input_val), uaf_torch(single_input_torch).item(), atol=1e-6)

    multi_input_val = np.array([float32(1.0), float32(0.5), float32(-0.5)], dtype=Float32)
    multi_input_torch = torch.tensor(multi_input_val, dtype=torch.float32)

    np_output = np.array([uaf(x) for x in multi_input_val], dtype=Float32)
    torch_output = uaf_torch(multi_input_torch).detach().cpu().numpy()

    assert np.allclose(np_output, torch_output, atol=1e-6)