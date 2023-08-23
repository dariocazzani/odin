import inspect
from collections import deque
from typing import Callable
import random

import graphviz

from .ops import sigmoid
from .ops import tanh
from .ops import relu
from .ops import identity

class Node:
    def __init__(self, label, activation:Callable, state:float=0., bias:float=0):
        self._label = label
        self._state = state
        self._bias = bias
        self._pre_fire:float = 0
        self._activation = activation
        self._validate_activation()
        self._outgoing_edges = []

    def _validate_activation(self):
        if not callable(self._activation):
            raise ValueError("activation must be a callable function")

        sig = inspect.signature(self._activation)
        if len(sig.parameters) != 1:
            raise ValueError("activation must be a unary function")
        try:
            result = self._activation(1.0)
            if not isinstance(result, float):
                raise ValueError("activation must return a float")
        except Exception as exc:
            raise ValueError(f'activation must be able to handle float inputs\n{exc}') from exc

    @property
    def label(self):
        return self._label

    @property
    def state(self):
        return self._state
    
    @property
    def bias(self):
        return self._bias

    @state.setter
    def state(self, value):
        self._state = value

    @bias.setter
    def bias(self, value):
        self._bias = value
        
    def add_edge(self, edge):
        self._outgoing_edges.append(edge)

    def add_input(self, value:float):
        self._pre_fire += value

    # Fire!
    def compute_activation(self):
        self._state += self._pre_fire
        self._state += self.bias
        self._state = self._activation(self._state)
        self._pre_fire = 0.

    @property
    def outgoing_edges(self):
        return self._outgoing_edges

    def __str__(self):
        return f"Node(label={self._label}, state={self._state}, bias={self._bias})"


class Edge:
    def __init__(self, start_node, end_node, weight):
        self._start_node = start_node
        self._end_node = end_node
        self._weight = weight
        self._start_node.add_edge(self)
        self._id = f"{start_node.label}_{end_node.label}"  # Unique label for each edge
        
    @property
    def id(self):
        return self._id
    
    @property
    def start_node(self):
        return self._start_node

    @property
    def end_node(self):
        return self._end_node

    @property
    def weight(self):
        return self._weight

    def __str__(self):
        return f"Edge(start={self._start_node.label}, end={self._end_node.label}, weight={self._weight})"


class SphericalEngine:
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

        self._adjacency_dict = adjacency_dict
        self._activations = activations
        self._biases = biases
        self._output_node_ids = output_node_ids
        self._input_node_ids = input_node_ids
        self._stateful = stateful
        self._max_steps = max_steps
        
        self._validate_inputs()

        self._nodes = {label: Node(label, activation=act) for label, act in activations.items()}
        self._edges = []
        self._current_queue = deque()
        self._next_queue = deque()
        self._active_nodes = deque()
        self._energy:int | None = None

        # biases
        for node_id, value in biases.items():
            if node_id in self._nodes:
                self._nodes[node_id].bias = value
    
        # Edges with weights
        for start_label, connections in adjacency_dict.items():
            for end_label, weight in connections.items():
                self._edges.append(Edge(self._nodes[start_label], self._nodes[end_label], weight))


    def _validate_inputs(self) -> None:
        """Validates input data for the Graph class."""
        adjacency_keys = set(self._adjacency_dict.keys())

        if not set(self._activations.keys()) == adjacency_keys:
            raise ValueError("Mismatch between activations and adjacency_dict keys.")
        if not set(self._biases.keys()) == adjacency_keys:
            raise ValueError("Mismatch between biases and adjacency_dict keys.")
        if not set(self._input_node_ids).issubset(adjacency_keys):
            raise ValueError("Some input_node_ids are not in the adjacency_dict keys.")
        if not set(self._output_node_ids).issubset(adjacency_keys):
            raise ValueError("Some output_node_ids are not in the adjacency_dict keys.")

    
    @property
    def input_node_ids(self) -> set:
        return self._input_node_ids
    
    
    @property
    def output_node_ids(self) -> set:
        return self._output_node_ids
    
    
    @property
    def activations(self) -> dict:
        return self._activations
    
    
    def _ensure_energy(self) -> None:
        """Ensure that the energy is set by running a dummy inference if necessary."""
        if self._energy is None:
            # Dummy inference to populate energy
            input_values = {key: random.random() for key in self.input_node_ids}
            self.inference(input_values=input_values)
            self.reset()


    @property
    def energy(self) -> int:
        self._ensure_energy()
        assert self._energy is not None
        return self._energy
    
    
    def add_input_ids(self, new_ids: set[int]) -> bool:
        """
        Adds new input node IDs to the Graph.

        Args:
        - new_ids (set[int]): The set of new input node IDs.

        Returns:
        - bool: True if successfully added, False otherwise.
        """
        adjacency_keys = set(self._adjacency_dict.keys())
        if not new_ids.issubset(adjacency_keys):
            return False
        self._input_node_ids.update(new_ids)
        return True
    

    def add_output_ids(self, new_ids: set[int]) -> bool:
        """
        Adds new output node IDs to the Graph.

        Args:
        - new_ids (set[int]): The set of new output node IDs.

        Returns:
        - bool: True if successfully added, False otherwise.
        """
        adjacency_keys = set(self._adjacency_dict.keys())
        if not new_ids.issubset(adjacency_keys):
            return False
        self._output_node_ids.update(new_ids)
        return True


    def reset(self):
        for node in self._nodes.values():
            node.state = 0
        self._current_queue = deque()
        self._next_queue = deque()
        self._active_nodes = deque()


    def inference(self, input_values:dict, verbose:bool=False) -> dict:
        input_values = {self._nodes[label]: value for label, value in input_values.items()}
        # Create source edges for each input node
        source_edges = [Edge(Node("Source", state=value, activation=lambda x: x), node, weight=1) for node, value in input_values.items()]

        # Enqueue the source edges
        self._current_queue.extend(source_edges)
        self._next_queue = deque()
        # Initialize deque for nodes that received input
        self._active_nodes.extend([node for node, _ in input_values.items()])
        # First step is a dummy step to run the inputs into the input nodes
        step = -1
        # Input edges don't count
        energy_used:int = 0 - len(source_edges)

        while self._current_queue and step < self._max_steps:
            traversed_edges = set()  # Set to store traversed edges
            energy_used += len(self._current_queue)
            while self._current_queue:
                edge = self._current_queue.popleft()
                input_value = edge.start_node.state * edge.weight
                edge.end_node.add_input(input_value)

                # Add node to active_nodes
                if edge.end_node not in self._active_nodes:
                    self._active_nodes.append(edge.end_node)

                # Add only the outgoing edges that have not been traversed yet
                for out_edge in edge.end_node.outgoing_edges:
                    if out_edge.id not in traversed_edges:
                        self._next_queue.append(out_edge)
                        traversed_edges.add(out_edge.id) # Mark the edge as traversed

            # Apply activation to all active nodes
            while self._active_nodes:
                node = self._active_nodes.popleft()
                node.compute_activation()

            self._current_queue, self._next_queue = self._next_queue, self._current_queue
            step += 1
            if verbose:
                print(f'Step {step}, current state: {[str(node) for node in self._nodes.values()]}')

        output_nodes = {nid: self._nodes[nid].state for nid in self._output_node_ids}

        self._energy = energy_used
        if verbose:
            print(f"Total energy used: {energy_used}")
        
        if not self._stateful:
            self.reset()
        
        return output_nodes


    def visualize(self, node_names:dict={}) -> None:
        dot = graphviz.Digraph()
        
        for node, attr in self.activations.items():
            node_name = node_names.get(node, "")
            activation_function = attr.__name__

            node_color:str = 'lightblue'
            if node in self.output_node_ids:
                node_color = 'lightgreen'
            if node in self.input_node_ids:
                node_color = 'orange'

            dot.node(f"{node}", label=f"{node}\nActivation: {activation_function}\n{node_name}", style='filled', fillcolor=node_color)

        for node, connections in self._adjacency_dict.items():
            for neighbor, weight in connections.items():
                dot.edge(f"{node}", f"{neighbor}", label=f"Weight: {weight:.2f}", arrowhead="normal", arrowtail="normal")
        
        for output_node in self._output_node_ids:
            dot.node(f"{output_node}", pos="2,0!") #type: ignore

        dot.format = 'png'
        file_path_without_extension = './graph_output'
        dot.render(file_path_without_extension, view=True)


    def get_pull_adjacency_dict(self) -> dict[int, dict[int, float]]:
        reversed_dict = {node: {} for node in self._adjacency_dict.keys()}
        
        for source_node, connections in self._adjacency_dict.items():
            for target_node, weight in connections.items():
                if target_node not in reversed_dict:
                    reversed_dict[target_node] = {}
                reversed_dict[target_node][source_node] = weight

        return reversed_dict

    

    # @staticmethod
    # def genes_to_adjacency(genome, config):
    #     adjacency_matrix = {}

    #     # Gather expressed connections.
    #     connections = [cg.key for cg in genome.connections.values()]# if cg.enabled]
    #     all_nodes = set([k for k, _ in  genome.nodes.items()] + config.genome_config.input_keys)
       
    #     biases = {}
    #     for node_idx, node in genome.nodes.items():
    #         biases[node_idx] = node.bias
        
    #     for node in all_nodes:
    #         edges = {}
    #         for conn_key in connections:
    #             inode, onode = conn_key
    #             if node == inode:
    #                 cg = genome.connections[conn_key]
    #                 edges[onode] = cg.weight

    #         if node in config.genome_config.input_keys:
    #             activation_function = identity
    #         else:
    #             ng = genome.nodes[node]
    #             activation_function = config.genome_config.activation_defs.get(ng.activation)
    #         adjacency_matrix[node] = {
    #             'activation': activation_function,
    #             'edges': edges
    #         }
    #     return adjacency_matrix, config.genome_config.output_keys, biases

    # @staticmethod
    # def create(genome, config):
    #     matrix, output_nodes, biases = Graph.genes_to_adjacency(genome, config)
    #     return Graph(matrix, output_nodes, biases=biases)



def main():
    # Define adjacency matrix
    """
        0     1     2     3     4     5     6     7     8   
    0  0.0   0.0   0.7   0.0   0.0   0.0   0.0   0.0   0.0 
    1  0.0   0.0  -0.5   0.9   0.0   0.0   0.0   0.0   0.0 
    2  0.0   0.0   0.0   0.0   0.8  -0.3   0.0   0.0   0.0 
    3  0.0   0.0   0.0   0.0   0.0   0.0  -0.4   0.0   0.0 
    4  0.0   0.0   0.0   0.6   0.0   0.0   0.0   0.0   0.1 
    5  0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.2   0.0 
    6  0.0   0.0   0.0   0.0   0.0   0.0   0.0  -0.9   0.0 
    7  0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0 
    8  0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0 
    """
    
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
    

    # list of tuples mapping node labels to input values
    input_values = {0: 1, 1: 0.5}

    graph = SphericalEngine(
        adjacency_dict=adjacency_dict,
        activations=activations,
        biases=biases,
        input_node_ids={0, 1},
        output_node_ids={7, 8},
        max_steps=6)
    
    print(f"Energy: {graph.energy}")

    result = graph.inference(input_values, verbose=False)
    graph.visualize()
    
    
    print(graph._adjacency_dict)
    print("=============")
    print(graph.get_pull_adjacency_dict())
    
    # output_nodes = result["output_nodes"]
    # energy = result["energy"]
    # print(f"Energy used: {energy:.2f}")
    # for node in output_nodes:
    #     print(f'Output Node {node.label} final state: {node.state:.4f}')
    
    # result = graph.inference(input_values)
    # output_nodes = result["output_nodes"]
    # energy = result["energy"]
    # print(f"Energy used: {energy:.2f}")
    # for node in output_nodes:
    #     print(f'Output Node {node.label} final state: {node.state:.4f}')
    # graph.reset()

    # result = graph.inference(input_values)
    # output_nodes = result["output_nodes"]
    # energy = result["energy"]
    # print(f"Energy used: {energy:.2f}")
    # for node in output_nodes:
    #     print(f'Output Node {node.label} final state: {node.state:.4f}')
    

if __name__ == "__main__":
    main()