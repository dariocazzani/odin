import numpy as np

from inference_engines.spherical import SphericalEngine
from inference_engines.ops import sigmoid
from inference_engines.ops import tanh
from inference_engines.ops import relu
from inference_engines.ops import identity

from interfaces.custom_types import AdjacencyDictType

TOL = 1E-6

def test_behavior_1() -> None:
    weight_0_1:np.float32 = np.float32(np.random.randn())
    input_a:np.float32 = np.float32(np.random.randn())
    activations = {0: sigmoid, 1: sigmoid}
    adjacency_dict:AdjacencyDictType = {
        0: {1: weight_0_1},
        1: {}
    }
    biases = {node: np.float32(0.) for node in adjacency_dict.keys()}
    input_values = {0: input_a}
    node_0_activation = activations.get(0, callable)
    node_1_activation = activations.get(1, callable)
    
    graph = SphericalEngine(
        adjacency_dict=adjacency_dict,
        activations=activations,
        biases=biases,
        output_node_ids={1},
        input_node_ids={0},
        stateful=True,
        max_steps=6)
    output_nodes = graph.inference(verbose=True, input_values=input_values)
    energy = graph.energy
    assert energy == 1
    assert abs(output_nodes.get(1, 0.) - node_1_activation(weight_0_1 * node_0_activation(input_a))) < TOL
    

def test_behavior_2() -> None:
    weights_vector = np.random.randn(3).astype(np.float32)
    inputs_vectors = np.random.randn(3).astype(np.float32)
    activations = {0: identity, 1: identity, 2: identity, 3: tanh}
    
    adjacency_dict:AdjacencyDictType = {
        0: {3: weights_vector[0]},
        1: {3: weights_vector[1]},
        2: {3: weights_vector[2]},
        3: {},
    }
    input_values = {
                    0: inputs_vectors[0],
                    1: inputs_vectors[1],
                    2: inputs_vectors[2]
    }
    biases = {node: np.float32(0.) for node in adjacency_dict.keys()}
    
    graph = SphericalEngine(
        adjacency_dict=adjacency_dict,
        activations=activations,
        biases=biases,
        output_node_ids={3},
        input_node_ids={0, 1, 2},
        stateful=True,
        max_steps=6)
    output_nodes = graph.inference(verbose=True, input_values=input_values)
    assert abs(output_nodes.get(3, 0.0) - tanh(np.dot(weights_vector, inputs_vectors))) < TOL
    
    
def test_behavior_3() -> None:
    weights_vector1 = np.random.randn(3)
    weights_vector2 = np.random.randn(2)
    weights_vector3 = np.random.randn(2)
    inputs_vectors = np.random.randn(3)
    
    activations = {0: identity, 1: identity, 2: identity, 3: relu, 4: relu, 5: tanh}
    adjacency_dict:AdjacencyDictType = {
        0: {3: weights_vector1[0]},
        1: {3: weights_vector1[1], 4: weights_vector2[0]},
        2: {3: weights_vector1[2], 4: weights_vector2[1]},
        3: {5: weights_vector3[0]},
        4: {5: weights_vector3[1]},
        5: {}
    }
    input_values = {
                    0: inputs_vectors[0],
                    1: inputs_vectors[1],
                    2: inputs_vectors[2]
    }
    biases = {node: np.float32(0.) for node in adjacency_dict.keys()}
    graph = SphericalEngine(
        adjacency_dict=adjacency_dict,
        activations=activations,
        biases=biases,
        input_node_ids={0, 1, 2},
        output_node_ids={5},
        stateful=True,
        max_steps=6
    )
    output_nodes = graph.inference(verbose=True, input_values=input_values)
    energy = graph.energy
    assert energy == 7
    output_value = output_nodes.get(5, 0.0)
    
    node_3_activation = activations.get(3, callable)
    node_4_activation = activations.get(4, callable)
    node_5_activation = activations.get(5, callable)
    
    layer_3 = node_3_activation(np.dot(inputs_vectors, weights_vector1))
    layer_4 = node_4_activation(np.dot(inputs_vectors[1:], weights_vector2))
    layer_3_4_vector = np.array([layer_3, layer_4])
    
    assert abs(output_value - node_5_activation(np.dot(layer_3_4_vector, weights_vector3))) < TOL
    

def test_behavior_4() -> None:
    weight_0_1 = np.float32(np.random.randn())
    weight_0_2 = np.float32(np.random.randn())
    weight_1_2 = np.float32(np.random.randn())
    weight_1_3 = np.float32(np.random.randn())
    weight_2_3 = np.float32(np.random.randn())
    
    activations = {0: tanh, 1: relu, 2: sigmoid, 3: identity}
    adjacency_dict:AdjacencyDictType = {
        0: {1: weight_0_1, 2: weight_0_2},
        1: {2: weight_1_2, 3: weight_1_3},
        2: {3: weight_2_3},
        3: {}
    }
    
    biases = {node: np.float32(0.) for node in adjacency_dict.keys()}
    input_values = {0: np.float32(np.random.randn())} 
    graph = SphericalEngine(
        adjacency_dict=adjacency_dict,
        activations=activations,
        biases=biases,
        input_node_ids={0},
        output_node_ids={3},
        stateful=True,
    )
    
    output_nodes = graph.inference(verbose=True, input_values=input_values)
    energy = graph.energy
    
    assert energy == 6
    
    node_0_activation = activations.get(0, callable)
    node_1_activation = activations.get(1, callable)
    node_2_activation = activations.get(2, callable)
    # Step 1
    b = node_1_activation(node_0_activation(input_values.get(0, np.float32(0))) * weight_0_1)
    c = node_2_activation(node_0_activation(input_values.get(0, np.float32(0))) * weight_0_2)
    # Step 2
    d = b * weight_1_3 + c * weight_2_3
    c = node_2_activation(c + b * weight_1_2)
    # Step 3
    d = d + c * weight_2_3
    output_value = output_nodes.get(3, 0.0)
    assert abs(d - output_value) < TOL


def test_behavior_5() -> None:
    weight_0_1 = np.float32(np.random.randn())
    activations = {0: identity, 1: identity}
    adjacency_dict:AdjacencyDictType = {
        0: {1: weight_0_1},
        1: {}
    }
    input_values = {0: np.float32(np.random.randn())}
    biases = {node: np.float32(0.) for node in adjacency_dict.keys()}
    graph = SphericalEngine(
        adjacency_dict=adjacency_dict,
        activations=activations,
        biases=biases,
        output_node_ids={1},
        input_node_ids={0},
        stateful=True,
        max_steps=6   
    )
    output_nodes = graph.inference(verbose=True, input_values=input_values)
    energy = graph.energy
    
    assert energy == 1
    
    input_val = input_values.get(0, np.float32(0.))
    node_0_activation = activations.get(0, callable)
    node_1_activation = activations.get(1, callable)
    state_0 = node_0_activation(input_val)
    state_1 = node_1_activation(state_0 * weight_0_1) 
    assert abs(output_nodes.get(1, 0.0) - state_1) < TOL
    
    # Traverse again without resetting
    output_nodes = graph.inference(verbose=True, input_values=input_values)
    state_0 = node_0_activation(state_0 + input_val)
    state_1 = node_1_activation(state_0 * weight_0_1 + state_1)
    assert abs(output_nodes.get(1, 0.0) - state_1) < TOL
    
    # Traverse again after resetting    
    graph.reset()
    output_nodes = graph.inference(verbose=True, input_values=input_values)
    state_0 = node_0_activation(input_val)
    state_1 = node_1_activation(state_0 * weight_0_1) 
    assert abs(output_nodes.get(1, 0.0) - state_1) < TOL
    
    
def test_behavior_6() -> None:
    weight_0_1 = np.float32(np.random.randn())
    weight_1_2 = np.float32(np.random.randn())
    weight_2_0 = np.float32(np.random.randn())
    activations = {0: sigmoid, 1: sigmoid, 2: sigmoid}
    adjacency_dict:AdjacencyDictType = {
        0: {1: weight_0_1},
        1: {2: weight_1_2},
        2: {0: weight_2_0}
    }
    input_values = {
        0: np.float32(np.random.randn()),
        1: np.float32(np.random.randn()),
        2: np.float32(np.random.randn()),
    }
    biases = {node: np.float32(0.) for node in adjacency_dict.keys()}
    graph = SphericalEngine(
        adjacency_dict=adjacency_dict,
        activations=activations,
        biases=biases,
        output_node_ids={0, 1, 2},
        input_node_ids={0, 1, 2},
        stateful=True,
        max_steps=1
    )
  
    graph.inference(verbose=True, input_values=input_values)
    energy = graph.energy
    assert energy == 3
    
    graph = SphericalEngine(
        adjacency_dict=adjacency_dict,
        activations=activations,
        biases=biases,
        output_node_ids={0, 1, 2},
        input_node_ids={0, 1, 2},
        stateful=True,
        max_steps=6
    )
    graph.inference(verbose=True, input_values=input_values)
    energy = graph.energy
    assert energy == 18
    
    NUM_STEPS:int = 6
    graph = SphericalEngine(
        adjacency_dict=adjacency_dict,
        activations=activations,
        biases=biases,
        output_node_ids={0, 1, 2},
        input_node_ids={0, 1, 2},
        stateful=True,
        max_steps=NUM_STEPS
    )
    output_nodes = graph.inference(input_values=input_values)
    
    input_0 = input_values.get(0, np.float32(0.0)) 
    input_1 = input_values.get(1, np.float32(0.0)) 
    input_2 = input_values.get(2, np.float32(0.0))
    node_0_activation = activations[0]
    node_1_activation = activations[1]
    node_2_activation = activations[2]
    
    next_step_a = next_step_b = next_step_c = np.float32(0)
    prev_step_a = node_0_activation(input_0)  
    prev_step_b = node_1_activation(input_1)  
    prev_step_c = node_2_activation(input_2)  

    for _ in range(NUM_STEPS):
        # Think of the signal from node X to node Y to need some time to propagate
        next_step_a = node_0_activation(prev_step_c * weight_2_0 + prev_step_a)
        next_step_b = node_1_activation(prev_step_a * weight_0_1 + prev_step_b)
        next_step_c = node_2_activation(prev_step_b * weight_1_2 + prev_step_c)
        prev_step_a = next_step_a
        prev_step_b = next_step_b
        prev_step_c = next_step_c
        
    assert abs(output_nodes.get(0, 0.0) - next_step_a) < TOL
    assert abs(output_nodes.get(1, 0.0) - next_step_b) < TOL
    assert abs(output_nodes.get(2, 0.0) - next_step_c) < TOL