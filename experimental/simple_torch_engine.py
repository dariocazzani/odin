import torch
import torch.nn as nn
from typing import Callable
import pretty_errors

from inference_engines.ops import sigmoid, relu, tanh, identity, step_fn
from inference_engines.ops import sigmoid_torch, relu_torch, tanh_torch, identity_torch, step_fn_torch
from inference_engines.ops import activation_mapping
from inference_engines.spherical import SphericalEngine

    
class DynamicSphericalTorch(nn.Module):
    def __init__(
        self,
        adjacency_dict:dict[int, dict[int, float]],
        activations:dict[int, Callable],
        biases:dict[int, float],
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
                layer.weight.data.fill_(weight)
                self._layers[layer_name] = layer
        
        # Initialize the state of each node of the network
        # TODO: make it possible to load states
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
        
    def _initialize_activations_torch(self, activations: dict[int, Callable], activation_mapping: dict[Callable, Callable]):
        self._activations_torch = {}
        for key, func in activations.items():
            if func in activation_mapping:
                self._activations_torch[key] = activation_mapping[func]
            else:
                raise ValueError(f"Unknown activation function: {func}")

    
    def forward(self, x) -> dict:
        # Loop through each input and apply the corresponding layer, bias, and activation
        for input_id in self._input_node_ids:
            slice_x = x[:, input_id].unsqueeze(1)  # Shape [batch_size, 1]
            activation = self._activations_torch.get(input_id, lambda x: x)
            bias = self._biases_torch.get(str(input_id), 0.0)
            
            output = self._layers[f"-1_{input_id}"](slice_x)
            output = activation(output + self._states[input_id] + bias + self._states[input_id])
            
            self._states[input_id] = output
        for step in range(0, self._max_steps):
            next_edge = self._edges_by_step.get(step, {})
            for end_node, source_nodes in next_edge.items():
                activation = self._activations_torch[end_node]
                bias = self._biases_torch[str(end_node)]
                outputs = []
                for source in source_nodes:
                    _input = self._states[source]
                    _output = self._layers[f"{source}_{end_node}"](_input)
                    outputs.append(_output)
                self._states[end_node] = activation(sum(outputs) + bias + self._states[end_node])
        
        return {node_id: self._states[node_id] for node_id in self._output_node_ids}        


def main():
    torch.random.manual_seed(53)
    
    biases = {0: 0.0, 1: 0.0, 2: 0.0, 3:0.0, 4:0.0, 5:0.0, 6:0.0, 7:0.0, 8:0.0}
    activations = {
        0: sigmoid,
        1: relu,
        2: tanh,
        3: sigmoid,
        4: tanh,
        5: identity,
        6: sigmoid,
        7: tanh,
        8: identity
    }
    
    
    adjacency_dict = {
        0: {2: 0.7},
        1: {2: -0.5, 3: 0.9},
        2: {4: 0.8, 5: -0.3},
        3: {6: -0.4},
        4: {3: 0.6, 8: 0.1},
        5: {7: 0.2},
        6: {7: -0.9},
        7: {},
        8: {},
    }
    input_nodes_ids = {0, 1}
    output_nodes_ids = {7, 8}
    
    batch_size = 10
    x = torch.randn(batch_size, len(input_nodes_ids))
    
    dynamic_engine = DynamicSphericalTorch(
        adjacency_dict=adjacency_dict,
        activations=activations,
        biases=biases,
        input_node_ids=input_nodes_ids,
        output_node_ids=output_nodes_ids,
        stateful=True
    )
    
    engine_out = dynamic_engine(x)

    graph = SphericalEngine(
        adjacency_dict=adjacency_dict,
        activations=activations,
        biases=biases,
        input_node_ids=input_nodes_ids,
        output_node_ids=output_nodes_ids,
        stateful=True
    )
    
    # graph.visualize()
    
    for idx, value in enumerate(x):
        input_values = {node_id: value[i].item() for i, node_id in enumerate(input_nodes_ids)}
        result = graph.inference(input_values=input_values, verbose=False)
        for key, value in result.items():
            assert abs(value - engine_out.get(key)[idx].detach().numpy()) < 1E-4
        print(f"Test for batch {idx+1} passed!")
        graph.reset()
        
if __name__ == "__main__":
    main()
