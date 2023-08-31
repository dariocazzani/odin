import numpy as np
from enum import Enum
import numpy as np

import torch
import torch.nn.functional as F


UAF_NUM_PARAMS = 5
UAF_PARAMS = ['A', 'B', 'C', 'D', 'E']


# PyTorch functions
def sigmoid_torch(x: torch.Tensor) -> torch.Tensor:
    x = x.to(torch.float32)
    return 1 / (1 + torch.exp(-x))

def relu_torch(x: torch.Tensor) -> torch.Tensor:
    x = x.to(torch.float32)
    return F.relu(x)

def tanh_torch(x: torch.Tensor) -> torch.Tensor:
    x = x.to(torch.float32)
    return torch.tanh(x)

def identity_torch(x: torch.Tensor) -> torch.Tensor:
    x = x.to(torch.float32)
    return x

def elu_torch(x: torch.Tensor) -> torch.Tensor:
    x = x.to(torch.float32)
    return F.elu(x)


# NumPy functions
def sigmoid(x: np.float32) -> np.float32:
    x = np.float32(x)
    return np.float32(1.0) / (np.float32(1.0) + np.exp(-x))


def relu(x: np.float32) -> np.float32:
    x = np.float32(x)
    return np.maximum(np.float32(0.), x)

def tanh(x: np.float32) -> np.float32:
    x = np.float32(x)
    return np.tanh(x)

def identity(x: np.float32) -> np.float32:
    x = np.float32(x)
    return x

def elu(x: np.float32) -> np.float32:
    x = np.float32(x)
    return np.float32(np.where(x > 0, x, np.exp(x) - 1))



activation_mapping = {
    sigmoid: sigmoid_torch,
    relu: relu_torch,
    tanh: tanh_torch,
    identity: identity_torch,
    elu: elu_torch,
}

class ActivationFunctions(Enum):
    RELU = "relu"
    SIGMOID = "sigmoid"
    TANH = "tanh"
    IDENTITY = "identity"
    ELU = "elu"

FUNCTION_MAP = {
    ActivationFunctions.RELU: relu,
    ActivationFunctions.SIGMOID: sigmoid,
    ActivationFunctions.TANH: tanh,
    ActivationFunctions.IDENTITY: identity,
    ActivationFunctions.ELU: elu,
}


class UAF:
    def __init__(self):
        self.A = 1.0
        self.B = 0.0
        self.C = 0.0
        self.D = -1.0
        self.E = 0.0

    def relu(self, x):
        return np.maximum(0, x)

    def perturbe_parameter(self, param_name: str, perturbation: float) -> None:
      param = getattr(self, param_name)
      setattr(self, param_name, param + perturbation)

    def __call__(self, input):
        P1 = (self.A * (input + self.B)) + (self.C * np.square(input))
        P2 = (self.D * (input - self.B))

        P3 = self.relu(P1) + np.log1p(np.exp(-np.abs(P1)))
        P4 = self.relu(P2) + np.log1p(np.exp(-np.abs(P2)))
        return P3 - P4  + self.E


def create_step_uaf():
    uaf = UAF()
    uaf.A = 1000.0
    uaf.B = 0.0005
    uaf.C = 0
    uaf.E = 0
    uaf.D = 1000.0
    return uaf


def create_sigmoid_uaf():
    uaf = UAF()
    uaf.A = 1.01605291
    uaf.B = 1 / (2 * uaf.A)
    uaf.C = 0
    uaf.E = 0
    uaf.D = uaf.A
    return uaf


def create_tanh_uaf():
    uaf = UAF()
    uaf.A = 2.12616013
    uaf.B = 1 / uaf.A
    uaf.C = 0.0
    uaf.D = uaf.A
    uaf.E = -1.0
    return uaf


def create_relu_uaf():
    uaf = UAF()
    uaf.A = 1000.0
    uaf.B = 0.0
    uaf.C = 0.0
    uaf.D = uaf.A - 1
    uaf.E = 0.0
    return uaf