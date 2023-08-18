import numpy as np
import math


UAF_NUM_PARAMS = 5
UAF_PARAMS = ['A', 'B', 'C', 'D', 'E']


# Define activations
def sigmoid(x:float) -> float:
    return 1 / (1 + math.exp(-x))

def relu(x:float) -> float:
    return max(0, x)

def tanh(x:float) -> float:
    return math.tanh(x)

def identity(x:float) -> float:
    return x

def step_fn(x: float) -> float:
    if x < 0:
        return 0
    elif x == 0:
        return 0.5
    else:
        return 1


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