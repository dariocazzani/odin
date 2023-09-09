from typing import Callable

import numpy as np

from optimizers.mutator import Mutator
from inference_engines.ops import identity
from interfaces.custom_types import AdjacencyDictType
from interfaces.custom_types import float32

TOL = 1e-5  

def create_sample_data() -> tuple:
    adj_dict:AdjacencyDictType = {0: {1: float32(0.5)}, 1: {2: float32(0.3)}, 2: {}}
    biases = {0: 0.2, 1: 0.3, 2: 0.4}
    activations:dict[int, Callable] = {0: identity, 1: identity, 2: identity}
    input_nodes = {0}
    output_nodes = {2}    
    return adj_dict, biases, activations, input_nodes, output_nodes


def test_modify_weights() -> None:
    adj_dict, _, _, _, _ = create_sample_data()
    Mutator.set_mutation_prob(1.0)
    modified_adj_dict = Mutator.modify_weights(adj_dict)
    assert set(modified_adj_dict.keys()) == set(adj_dict.keys())
    for node_id, connections in adj_dict.items():
        for target_node, original_weight in connections.items():
            assert abs(modified_adj_dict[node_id][target_node] - original_weight) > TOL


def test_modify_biases() -> None:
    _, biases, _, _, _ = create_sample_data()
    Mutator.set_mutation_prob(1.0)
    modified_biases = Mutator.modify_biases(biases)
    assert set(modified_biases.keys()) == set(biases.keys())
    for node, origina_bias in biases.items():
        assert abs(modified_biases[node] - origina_bias) > TOL


def test_modify_activations() -> None:
    _, _, activations, _, _ = create_sample_data()
    Mutator.set_mutation_prob(1.0)
    modified_activations = Mutator.modify_activations(activations)
    assert set(modified_activations.keys()) == set(activations.keys())
    for node, original_func in activations.items():
        assert modified_activations[node] in Mutator.available_activations
        assert modified_activations[node] != original_func
        

def test_add_node() -> None:
    adj_dict, biases, activations, _, _ = create_sample_data()
    Mutator.set_add_node_prob(1.0)
    modified_adj_dict, modified_biases, modified_activations = Mutator.add_node(adj_dict.copy(), biases.copy(), activations.copy())  
    assert len(modified_adj_dict) == len(adj_dict) + 1
    assert len(modified_biases) == len(biases) + 1
    assert len(modified_activations) == len(activations) + 1


def test_add_connection() -> None:
    adj_dict, _, _, _, _ = create_sample_data()
    Mutator.set_add_connection_prob(1.0)
    modified_adj_dict = Mutator.add_connection(adj_dict.copy())
    total_connections_initial = sum(len(targets) for targets in adj_dict.values())
    total_connections_modified = sum(len(targets) for targets in modified_adj_dict.values())
    assert total_connections_modified == total_connections_initial + 1


def test_remove_connection() -> None:
    adj_dict, _, _, _, _ = create_sample_data()
    Mutator.set_remove_connection_prob(1.0)
    modified_adj_dict = Mutator.remove_connection(adj_dict.copy())
    total_connections_initial = sum(len(targets) for targets in adj_dict.values())
    total_connections_modified = sum(len(targets) for targets in modified_adj_dict.values())
    assert total_connections_modified == total_connections_initial - 1


def test_remove_node() -> None:
    adj_dict, biases, activations, input_nodes, output_nodes = create_sample_data()
    Mutator.set_remove_node_prob(1.0)
    modified_adj_dict, modified_biases, modified_activations = Mutator.remove_node(adj_dict.copy(), biases.copy(), activations.copy(), input_nodes, output_nodes)    
    assert len(modified_adj_dict) == len(adj_dict) - 1
    assert len(modified_biases) == len(biases) - 1
    assert len(modified_activations) == len(activations) - 1
    assert all(node in modified_adj_dict for node in input_nodes)
    assert all(node in modified_adj_dict for node in output_nodes)
