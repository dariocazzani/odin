import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from interfaces.custom_types import float32, Float32

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
def sigmoid(x:Float32) -> Float32:
    x = float32(x)
    return float32(1.0) / (float32(1.0) + np.exp(-x))

def relu(x:Float32) -> Float32:
    x = float32(x)
    return np.maximum(float32(0.), x)

def tanh(x:Float32) -> Float32:
    x = float32(x)
    return np.tanh(x)

def identity(x:Float32) -> Float32:
    x = float32(x)
    return x

def elu(x: Float32) -> Float32:
    x = float32(x)
    result = np.where(x > 0, x, np.exp(x) - 1)
    if isinstance(result, np.ndarray) and result.size == 1:
        return Float32(result.item())
    else:
        return Float32(result)



class UAF:
    def __init__(self):
        self.A = Float32(1.0)
        self.B = Float32(0.0)
        self.C = Float32(0.0)
        self.D = Float32(-1.0)
        self.E = Float32(0.0)
        
    @property
    def __name__(self) -> str: return "UAF"

    def relu(self, x:Float32) -> Float32:
        return np.maximum(0, x)

    def perturbe_parameter(self, param_name: str, perturbation: Float32) -> None:
        param = getattr(self, param_name)
        setattr(self, param_name, param + perturbation)

    def __call__(self, input:Float32):
        P1 = (self.A * (input + self.B)) + (self.C * np.square(input))
        P2 = (self.D * (input - self.B))

        P3 = self.relu(P1) + np.log1p(np.exp(-np.abs(P1)))
        P4 = self.relu(P2) + np.log1p(np.exp(-np.abs(P2)))
        return P3 - P4  + self.E
    

class UAF_Torch(nn.Module):
    def __init__(self):
        super(UAF_Torch, self).__init__()
        self.A = nn.Parameter(torch.tensor(1.0))
        self.B = nn.Parameter(torch.tensor(0.0))
        self.C = nn.Parameter(torch.tensor(0.0))
        self.D = nn.Parameter(torch.tensor(-1.0))
        self.E = nn.Parameter(torch.tensor(0.0))

    def relu(self, x):
        return F.relu(x)

    def forward(self, input):
        P1 = (self.A * (input + self.B)) + (self.C * torch.square(input))
        P2 = (self.D * (input - self.B))
        
        P3 = self.relu(P1) + torch.log1p(torch.exp(-torch.abs(P1)))
        P4 = self.relu(P2) + torch.log1p(torch.exp(-torch.abs(P2)))
        return P3 - P4 + self.E
    
    def perturbe_parameter(self, param_name: str, perturbation):
        param = getattr(self, param_name)
        with torch.no_grad():
            param.add_(perturbation)
            

activation_mapping = {
    sigmoid: sigmoid_torch,
    relu: relu_torch,
    tanh: tanh_torch,
    identity: identity_torch,
    elu: elu_torch,
}


FUNCTION_MAP = {
    "relu": relu,
    "sigmoid": sigmoid,
    "tanh": tanh,
    "identity": identity,
    "elu": elu,
}