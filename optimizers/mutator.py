
import random
from typing import Callable

import numpy as np

from inference_engines.ops import FUNCTION_MAP
from interfaces.custom_types import AdjacencyDictType
from logger import ColoredLogger

log = ColoredLogger("Mutator").get_logger()

class Mutator:
    mutation_prob = 0.05
    weight_mutation_amount = 0.1
    add_node_prob = 0.1
    add_connection_prob = 0.1
    remove_connection_prob = 0.1
    remove_node_prob = 0.1
    available_activations = list(FUNCTION_MAP.values())

    @classmethod
    def set_mutation_prob(cls, mutation_prob: float):
        cls.mutation_prob = mutation_prob

    @classmethod
    def set_weight_mutation_amount(cls, weight_mutation_amount: float):
        cls.weight_mutation_amount = weight_mutation_amount

    @classmethod
    def set_add_node_prob(cls, prob: float):
        cls.add_node_prob = prob

    @classmethod
    def set_add_connection_prob(cls, prob: float):
        cls.add_connection_prob = prob

    @classmethod
    def set_remove_connection_prob(cls, prob: float):
        cls.remove_connection_prob = prob

    @classmethod
    def set_remove_node_prob(cls, prob: float):
        cls.remove_node_prob = prob
    
        
    @staticmethod
    def modify_weights(adjacency_dict: AdjacencyDictType) -> AdjacencyDictType:
        modified_dict = {key: value.copy() for key, value in adjacency_dict.items()}
        for node_id, connections in modified_dict.items():
            for target_node, _ in connections.items():
                if random.random() < Mutator.mutation_prob:
                    modified_dict[node_id][target_node] += random.uniform(-Mutator.weight_mutation_amount, Mutator.weight_mutation_amount)
                    modified_dict[node_id][target_node] = max(-1, min(1, modified_dict[node_id][target_node])) #type: ignore
        return modified_dict


    @staticmethod
    def modify_biases(biases: dict[int, np.float32]) -> dict[int, np.float32]:
        modified_biases = biases.copy()
        for node_id in modified_biases:
            if random.random() < Mutator.mutation_prob:
                modified_biases[node_id] += random.uniform(-Mutator.weight_mutation_amount, Mutator.weight_mutation_amount)        
        return modified_biases


    @staticmethod
    def modify_activations(activations: dict[int, Callable]) -> dict[int, Callable]:
        new_activations = {}
        for node_id, func in activations.items():
            if random.random() < Mutator.mutation_prob:
                new_activations[node_id] = random.choice([f for f in Mutator.available_activations if f != func])
            else:
                new_activations[node_id] = activations[node_id]
        return new_activations
    
    @staticmethod
    def add_node(adjacency_dict: AdjacencyDictType, biases: dict[int, np.float32], activations: dict[int, Callable]) -> tuple:
        modified_adjacency_dict = {k: v.copy() for k, v in adjacency_dict.items()}
        modified_biases = biases.copy()
        modified_activations = activations.copy()

        if random.random() > Mutator.add_node_prob:
            return modified_adjacency_dict, modified_biases, modified_activations

        new_node_id: int = max(modified_adjacency_dict.keys()) + 1
        all_nodes: list[int] = list(modified_adjacency_dict.keys())

        input_node = np.random.choice(all_nodes)
        all_nodes.remove(input_node)
        output_node = np.random.choice(all_nodes)

        if input_node not in modified_adjacency_dict:
            modified_adjacency_dict[input_node] = {}
        modified_adjacency_dict[input_node][new_node_id] = np.float32(np.random.randn())

        if new_node_id not in modified_adjacency_dict:
            modified_adjacency_dict[new_node_id] = {}
        modified_adjacency_dict[new_node_id][output_node] = np.float32(np.random.randn())

        modified_biases[new_node_id] = np.float32(0.0)
        modified_activations[new_node_id] = random.choice(Mutator.available_activations)

        return modified_adjacency_dict, modified_biases, modified_activations


    @staticmethod
    def add_connection(adjacency_dict: AdjacencyDictType) -> AdjacencyDictType:
        modified_adjacency_dict = {k: v.copy() for k, v in adjacency_dict.items()}
        if random.random() > Mutator.add_connection_prob:
            return modified_adjacency_dict
        all_nodes: list[int] = list(modified_adjacency_dict.keys())
        random.shuffle(all_nodes)
        for node_id in all_nodes:
            potential_targets = [n for n in all_nodes if n not in modified_adjacency_dict[node_id]]
            if potential_targets:
                target_node = np.random.choice(potential_targets)
                modified_adjacency_dict[node_id][target_node] = np.float32(np.random.randn())
                break
        return modified_adjacency_dict

    
    @staticmethod
    def remove_connection(adjacency_dict: AdjacencyDictType) -> AdjacencyDictType:
        modified_adjacency_dict = {k: v.copy() for k, v in adjacency_dict.items()}
        if random.random() > Mutator.remove_connection_prob:
            return modified_adjacency_dict
        all_nodes: list[int] = list(modified_adjacency_dict.keys())
        nodes_with_connections = [node_id for node_id in all_nodes if len(modified_adjacency_dict[node_id]) > 0]
        if not nodes_with_connections:
            return modified_adjacency_dict
        node_id = np.random.choice(nodes_with_connections)
        target_node = np.random.choice(list(modified_adjacency_dict[node_id].keys()))
        del modified_adjacency_dict[node_id][target_node]
        return modified_adjacency_dict


    @staticmethod
    def remove_node(adjacency_dict: AdjacencyDictType, biases: dict[int, np.float32], activations: dict[int, Callable], input_node_ids: set[int], output_node_ids: set[int]) -> tuple:
        modified_adjacency_dict = {k: v.copy() for k, v in adjacency_dict.items()}
        modified_biases = biases.copy()
        modified_activations = activations.copy()

        if random.random() > Mutator.remove_node_prob:
            return modified_adjacency_dict, modified_biases, modified_activations

        all_nodes: list[int] = list(modified_adjacency_dict.keys())
        nodes_removable = [node_id for node_id in all_nodes if node_id not in input_node_ids and node_id not in output_node_ids]
        if len(nodes_removable) < 1:
            return modified_adjacency_dict, modified_biases, modified_activations

        node_to_remove = np.random.choice(nodes_removable)

        if node_to_remove in modified_adjacency_dict:
            del modified_adjacency_dict[node_to_remove]
        for _, connections in modified_adjacency_dict.items():
            if node_to_remove in connections:
                del connections[node_to_remove]

        if node_to_remove in modified_biases:
            del modified_biases[node_to_remove]

        if node_to_remove in modified_activations:
            del modified_activations[node_to_remove]

        return modified_adjacency_dict, modified_biases, modified_activations

