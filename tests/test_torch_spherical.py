import os
import random

import torch
import numpy as np

from inference_engines.torch_spherical import DynamicSphericalTorch, SphericalEngine
from inference_engines.ops import FUNCTION_MAP
from logger import ColoredLogger
from interfaces.custom_types import AdjacencyDictType

log = ColoredLogger(os.path.basename(__file__)).get_logger()
torch.random.manual_seed(53)
np.random.seed(53)
random.seed(53)

def generate_parameters() -> tuple:
    num_inputs:int = random.randint(1, 5)
    num_outputs:int = random.randint(1, 5)
    num_hiddens:int = random.randint(1, 5)

    input_nodes_ids = set(range(0, num_inputs))    
    output_nodes_ids = set(range(num_inputs, num_inputs+num_outputs))
    hidden_node_ids = set(range(num_inputs + num_outputs, num_outputs + num_inputs + num_hiddens))    
    
    # TODO: <2023-08-28> Create interface for adjacency_dict
    adjacency_dict:AdjacencyDictType = {}
    all_nodes = input_nodes_ids.union(hidden_node_ids).union(output_nodes_ids)
    
    for node in all_nodes:
        if node in output_nodes_ids:
            adjacency_dict[node] = {}
            continue
        target_nodes = all_nodes - input_nodes_ids
        target_nodes_subset = random.sample(list(target_nodes), k=random.randint(0, len(target_nodes)))
        adjacency_dict[node] = {target: np.float32(np.random.randn() * 0.1) for target in target_nodes_subset}

    biases = {node: np.random.randn() for node in all_nodes}
    
    available_activations = list(FUNCTION_MAP.values())
    activations = {node: random.choice(available_activations) for node in all_nodes}
    
    return adjacency_dict, biases, activations, input_nodes_ids, output_nodes_ids


def test_dynamic_spherical_torch():
    adjacency_dict, biases, activations, input_nodes_ids, output_nodes_ids =generate_parameters()
    
    TOL = 1E-5
    max_steps = 6
    batch_size:int = 64
 
    x = torch.randn(batch_size, len(input_nodes_ids), dtype=torch.float32)
    
    dynamic_engine = DynamicSphericalTorch(
        adjacency_dict=adjacency_dict,
        activations=activations,
        biases=biases,
        input_node_ids=input_nodes_ids,
        output_node_ids=output_nodes_ids,
        stateful=True,
        max_steps=max_steps
    )
    
    engine_out = dynamic_engine(x)

    graph = SphericalEngine(
        adjacency_dict=adjacency_dict,
        activations=activations,
        biases=biases,
        input_node_ids=input_nodes_ids,
        output_node_ids=output_nodes_ids,
        stateful=True,
        max_steps=max_steps
    )
    
    for idx, value in enumerate(x):
        input_values = {node_id: value[i].item() for i, node_id in enumerate(input_nodes_ids)}
        result = graph.inference(input_values=input_values)
        for key, _value in result.items():
            assert abs(_value - engine_out.get(key)[idx].detach().numpy()) < TOL, \
                f"Assertion failed for batch {idx+1}, key {key}. Graph result: {_value}, Engine result: {engine_out.get(key)[idx].detach().numpy()}"
        log.info (f"Test for batch {idx+1} passed!")
        graph.reset()