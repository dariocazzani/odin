import numpy as np
from typing import Callable

Float32 = np.float32
AdjacencyDictType = dict[int, dict[int, Float32]]
BiasesType = dict[int, Float32]
ActivationsType = dict[int, Callable]

def float32(x: float | Float32) -> Float32: return np.float32(x)