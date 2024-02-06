import inspect
from typing import Callable

from odin.interfaces.custom_types import (
    Float32,
    float32
)


class Node:
    def __init__(
            self,
            label:int,
            activation:Callable,
            state=float32(0.),
            bias=float32(0.)
    ) -> None:

        self._label:int = label
        self._state:Float32 = state
        self._bias:Float32 = bias
        self._pre_fire:Float32 = float32(0.)
        self._activation:Callable = activation
        self._validate_activation()
        self._outgoing_edges:list[Edge] = []


    def _validate_activation(self):
        if not callable(self._activation):
            raise ValueError("activation must be a callable function")

        sig = inspect.signature(self._activation)
        if len(sig.parameters) != 1:
            raise ValueError("activation must be a unary function")
        try:
            result = self._activation(float32(1.0))
            if not result.dtype.name == 'float32':
                raise ValueError("activation must return a float32")
        except Exception as exc:
            raise ValueError(f'activation must be able to handle float inputs\n{exc}') from exc

    @property
    def label(self): return self._label

    @property
    def state(self): return self._state

    def set_state(self, value): self._state = value

    @property
    def bias(self): return self._bias

    def set_bias(self, value): self._bias = value

    def add_edge(self, edge): self._outgoing_edges.append(edge)

    def add_input(self, value:Float32): self._pre_fire += value

    @property
    def outgoing_edges(self):
        return self._outgoing_edges

    # Fire!
    def compute_activation(self):
        self._state += self._pre_fire
        self._state += self.bias
        self._state = self._activation(self._state)
        self._pre_fire = float32(0.)


    def __str__(self):
        return f"Node(label={self._label}, state={self._state:.5f}, bias={self._bias:.5f})"


class Edge:
    def __init__(self, start_node:Node, end_node:Node, weight:Float32):
        self._start_node = start_node
        self._end_node = end_node
        self._weight = weight
        self._start_node.add_edge(self)
        self._id = f"{start_node.label}_{end_node.label}"  # Unique label for each edge

    @property
    def id(self): return self._id

    @property
    def start_node(self): return self._start_node

    @property
    def end_node(self): return self._end_node

    @property
    def weight(self) -> Float32: return self._weight

    def __str__(self):
        return f"Edge(start={self._start_node.label}, end={self._end_node.label}, weight={self._weight})"
