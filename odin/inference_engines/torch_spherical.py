from typing import Callable

import torch
import torch.nn as nn
from odin.inference_engines.ops import activation_mapping
from odin.inference_engines.spherical import SphericalEngine
from odin.interfaces.custom_types import (
    ActivationsType,
    AdjacencyDictType,
    BiasesType
)


class DynamicSphericalTorch(nn.Module):
    def __init__(
        self,
        adjacency_dict:AdjacencyDictType,
        activations:ActivationsType,
        biases:BiasesType,
        output_node_ids:set[int],
        input_node_ids:set[int],
        stateful:bool,
        max_steps:int=6,
    ) -> None:

        super(DynamicSphericalTorch, self).__init__()

        self._adjacency_dict = adjacency_dict
        self._activations_numpy = activations
        self._biases_numpy = biases
        self._output_node_ids = output_node_ids
        self._input_node_ids = input_node_ids
        self._stateful = stateful
        self._max_steps = max_steps

        self._initialize_activations_torch(activations, activation_mapping)

        # Compile operations
        self._spherical_engine = self._init_spherical_engine()
        self._edges_by_step, self._energy = self._spherical_engine.compile()

        self._layers = nn.ModuleDict()
        self._biases_torch = nn.ParameterDict()
        self._states = {}

        # Create input connections:
        for node_id in input_node_ids:
            layer_name = f"{-1}_{node_id}"
            layer = nn.Linear(1, 1, bias=False)
            layer.weight.data.fill_(1.0)
            self._layers[layer_name] = layer

        # Create biases
        for node_id, bias_value in biases.items():
            self._biases_torch[str(node_id)] = nn.Parameter(torch.Tensor([bias_value]), requires_grad=True)

        # Create layers
        for node_id, connections in adjacency_dict.items():
            for target_node, weight in connections.items():
                layer_name = f"{node_id}_{target_node}"
                layer = nn.Linear(1, 1, bias=False)
                layer.weight.data.fill_(float(weight))
                self._layers[layer_name] = layer

        # Initialize the state of each node of the network
        # TODO: <2023-08-26> make it possible to load states
        for node_id in adjacency_dict.keys():
            self._states[node_id] = torch.zeros(1,1)


    def _init_spherical_engine(self) -> SphericalEngine:
        return SphericalEngine(
            adjacency_dict=self._adjacency_dict,
            activations=self._activations_numpy,
            biases=self._biases_numpy,
            input_node_ids=self._input_node_ids,
            output_node_ids=self._output_node_ids,
            stateful=self._stateful,
            max_steps=self._max_steps
        )

    def _initialize_activations_torch(self, activations: ActivationsType, activation_mapping: dict[Callable, Callable]):
        self._activations_torch = {}
        for key, func in activations.items():
            if func in activation_mapping:
                self._activations_torch[key] = activation_mapping[func]
            else:
                raise ValueError(f"Unknown activation function: {func}")


    def reset(self):
        for node_id in self._adjacency_dict.keys():
            self._states[node_id] = torch.zeros(1,1)


    def forward(self, x:torch.Tensor) -> dict:
        # Loop through each input and apply the corresponding layer, bias, and activation
        for input_id in self._input_node_ids:
            slice_x = x[:, input_id].unsqueeze(1)  # Shape [batch_size, 1]
            activation = self._activations_torch.get(input_id, lambda x: x)
            bias = self._biases_torch.get(str(input_id), 0.0)

            output = self._layers[f"-1_{input_id}"](slice_x)
            output = activation(output + bias + self._states[input_id])

            self._states[input_id] = output
        for step, next_edge in self._edges_by_step.items():
            if step < 0: continue
            temp_states:dict[int, torch.Tensor] = {}
            for end_node, source_nodes in next_edge.items():
                activation = self._activations_torch[end_node]
                bias = self._biases_torch.get(str(end_node), 0.0)
                outputs = []
                for source in source_nodes:
                    _input = self._states[source]
                    _output = self._layers[f"{source}_{end_node}"](_input)
                    outputs.append(_output)
                temp_states[end_node] = activation(sum(outputs) + bias + self._states[end_node])
            for end_node, result in temp_states.items():
                self._states[end_node] = result
        return {node_id: self._states[node_id] for node_id in self._output_node_ids}