import random
from collections import defaultdict, deque

import graphviz  #type: ignore
from odin.inference_engines.graph_elements import (
    Edge,
    Node
)
from odin.inference_engines.ops import (
    FUNCTION_MAP,
    elu,
    identity,
    relu,
    sigmoid,
    tanh
)
from odin.interfaces.custom_types import (
    ActivationsType,
    AdjacencyDictType,
    BiasesType,
    float32,
)


class SphericalEngine:
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

        self._adjacency_dict = adjacency_dict
        self._activations:ActivationsType = activations
        self._biases = biases
        self._output_node_ids = output_node_ids
        self._input_node_ids = input_node_ids
        self._stateful = stateful
        self._max_steps = max_steps

        self._validate_inputs()

        # Variables used for compilations
        self._queues_by_step:dict[int, deque] = {}

        self._nodes = {label: Node(label, activation=act) for label, act in activations.items()}
        self._edges = []
        self._current_queue:deque[Edge] = deque()
        self._next_queue:deque[Edge] = deque()
        self._active_nodes:deque[Node] = deque()
        self._energy:int | None = None

        for node_id, value in biases.items():
            if node_id in self._nodes:
                self._nodes[node_id].set_bias(value)
        for start_label, connections in adjacency_dict.items():
            for end_label, weight in connections.items():
                self._edges.append(Edge(self._nodes[start_label], self._nodes[end_label], weight))


    @classmethod
    def from_node_ids(cls, input_node_ids: set[int], output_node_ids: set[int]) -> 'SphericalEngine':
        # Assign random connections between inputs and outputs
        adjacency_dict: AdjacencyDictType = {}
        input_size = len(input_node_ids)
        output_size = len(output_node_ids)

        for i in range(input_size):
            potential_output_nodes = range(input_size, input_size + output_size)
            num_connections = random.randint(1, output_size)
            selected_output_nodes = random.sample(potential_output_nodes, num_connections)
            adjacency_dict[i] = {output: float32(random.uniform(-1, 1)) for output in selected_output_nodes}

        for i in range(input_size, input_size + output_size):
            adjacency_dict[i] = {}

        biases: BiasesType = {node: float32(0.0) for node in adjacency_dict.keys()}
        available_activations = list(FUNCTION_MAP.values())
        activations: ActivationsType = {node: random.choice(available_activations) for node in adjacency_dict.keys()}

        return cls(
            adjacency_dict=adjacency_dict,
            activations=activations,
            biases=biases,
            output_node_ids=output_node_ids,
            input_node_ids=input_node_ids,
            stateful=False,
            max_steps=6
        )


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
    def input_node_ids(self) -> set: return self._input_node_ids

    @property
    def output_node_ids(self) -> set: return self._output_node_ids

    @property
    def activations(self) -> dict: return self._activations

    @property
    def queues_by_step(self) -> dict: return self._queues_by_step


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
        adjacency_keys = set(self._adjacency_dict.keys())
        if not new_ids.issubset(adjacency_keys):
            return False
        self._input_node_ids.update(new_ids)
        return True


    def add_output_ids(self, new_ids: set[int]) -> bool:
        adjacency_keys = set(self._adjacency_dict.keys())
        if not new_ids.issubset(adjacency_keys):
            return False
        self._output_node_ids.update(new_ids)
        return True


    def reset(self):
        for node in self._nodes.values():
            node.set_state(float32(0))
        self._current_queue.clear()
        self._next_queue.clear()
        self._active_nodes.clear()


    def compile(self) -> tuple[dict, int]:
        self.inference(input_values={key: random.uniform(0, 1) for key in self._input_node_ids})
        edges_by_step = self.queues_by_step

        # for each step we have the list of source nodes going to each end_node
        grouped_by_end_node:dict[int, dict] = {}
        for step, edges in edges_by_step.items():
            grouped = defaultdict(list)
            for edge in edges:
                grouped[edge.end_node.label].append(edge.start_node.label)
            grouped_by_end_node[step] = dict(grouped)

        return grouped_by_end_node, self.energy


    def inference(self, input_values:dict, verbose:bool=False) -> dict:
        input_values = {self._nodes[label]: float32(value) for label, value in input_values.items()}  # Explicitly set as float32
        # Create source edges for each input node
        source_edges = [Edge(Node(-1, state=value, activation=lambda x: x), node, weight=float32(1.0)) for node, value in input_values.items()]

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
            self._queues_by_step[step] = self._current_queue.copy()
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
            if verbose:
                print(f'======= Step {step} Spherical =======')
                for node in self._nodes.values():
                    print(node)
            step += 1

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
            bias = self._biases[node]

            node_color:str = 'lightblue'
            if node in self.output_node_ids:
                node_color = 'lightgreen'
            if node in self.input_node_ids:
                node_color = 'orange'

            if node_name != "":
                node_name = f"Name: {node_name}"
            dot.node(
                f"{node}",
                label = f"ID = {node}\nActivation: {activation_function}\nbias: {bias:.4f}\n{node_name}",
                style='filled',
                fillcolor=node_color
            )

        for node, connections in self._adjacency_dict.items():
            for neighbor, weight in connections.items():
                dot.edge(f"{node}", f"{neighbor}", label=f"Weight: {weight:.2f}", arrowhead="normal", arrowtail="normal")

        for output_node in self._output_node_ids:
            dot.node(f"{output_node}", pos="2,0!") #type: ignore

        dot.format = 'png'
        file_path_without_extension = './graph_output'
        dot.render(file_path_without_extension, view=True)


    def get_pull_adjacency_dict(self) -> AdjacencyDictType:
        reversed_dict:AdjacencyDictType = {node: {} for node in self._adjacency_dict.keys()}

        for source_node, connections in self._adjacency_dict.items():
            for target_node, weight in connections.items():
                if target_node not in reversed_dict:
                    reversed_dict[target_node] = {}
                reversed_dict[target_node][source_node] = weight

        return reversed_dict


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

    activations = {
        0: sigmoid,
        1: relu,
        2: tanh,
        3: sigmoid,
        4: elu,
        5: identity,
        6: sigmoid,
        7: tanh,
        8: elu,
    }


    adjacency_dict: AdjacencyDictType = {
        0: {2: float32(0.7)},
        1: {2: float32(-0.5), 3: float32(0.9)},
        2: {4: float32(0.8), 5: float32(-0.3)},
        3: {6: float32(-0.4)},
        4: {3: float32(0.6), 8: float32(0.1)},
        5: {7: float32(0.2)},
        6: {7: float32(-0.9)},
        7: {},
        8: {},
    }
    biases = {node: float32(0.) for node in adjacency_dict.keys()}

    graph = SphericalEngine(
        adjacency_dict,
        activations,
        biases = biases,
        input_node_ids={0, 1},
        output_node_ids={7,8},
        stateful=True,
    )

    graph.visualize(node_names={0: "X", 1: "Y"})

if __name__ == "__main__":
    main()