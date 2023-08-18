from typing import cast
import numpy as np

from inference_engines.spherical import SphericalEngine
from inference_engines.spherical import Node
from inference_engines.ops import sigmoid
from inference_engines.ops import tanh
from inference_engines.ops import relu
from inference_engines.ops import identity

TOL = 1E-10

def test_behavior_1():
    weight_ab:float = np.random.randn()
    input_a:float = np.random.randn()
    adjacency_matrix = {
        'A': {'activation': sigmoid, 'edges': {'B': weight_ab}},
        'B': {'activation': sigmoid, 'edges': {}}
    }
    input_values = {'A': input_a}
    node_a_activation = adjacency_matrix.get('A', {}).get('activation', callable)
    node_b_activation = adjacency_matrix.get('B', {}).get('activation', callable)
    
    graph = SphericalEngine(adjacency_dict=adjacency_matrix , max_steps=6)
    result = graph.inference(verbose=True, input_values=input_values)
    output_nodes = result["output_nodes"]
    energy = result["energy_used"]
    assert energy == 1./2.
    assert abs(output_nodes[0].state - node_b_activation(weight_ab * node_a_activation(input_a))) < TOL
    

def test_behavior_2():
    weights_vector = np.random.randn(3)
    inputs_vectors = np.random.randn(3)
    adjacency_matrix = {
        'A': {'activation': identity, 'edges': {'D': weights_vector[0]}},
        'B': {'activation': identity, 'edges': {'D': weights_vector[1]}},
        'C': {'activation': identity, 'edges': {'D': weights_vector[2]}},
        'D': {'activation': tanh, 'edges': {}}
    }
    input_values = {
                    'A': inputs_vectors[0],
                    'B': inputs_vectors[1],
                    'C': inputs_vectors[2]
    }
    graph = SphericalEngine(adjacency_dict=adjacency_matrix, max_steps=6)
    result = graph.inference(verbose=True, input_values=input_values)
    output_nodes = result["output_nodes"]
    output_value = output_nodes[0].state
    assert abs(output_value - tanh(np.dot(weights_vector, inputs_vectors))) < TOL
    
    
def test_behavior_3():
    weights_vector1 = np.random.randn(3)
    weights_vector2 = np.random.randn(2)
    weights_vector3 = np.random.randn(2)
    inputs_vectors = np.random.randn(3)
    adjacency_matrix = {
        'A': {'activation': identity, 'edges': {'D': weights_vector1[0]}},
        'B': {'activation': identity, 'edges': {
                                                'D': weights_vector1[1],
                                                'E': weights_vector2[0]
                                                }},
        'C': {'activation': identity, 'edges': {
                                                'D': weights_vector1[2],
                                                'E': weights_vector2[1]
                                                }},
        'D': {'activation': relu, 'edges': {'F': weights_vector3[0]}},
        'E': {'activation': relu, 'edges': {'F': weights_vector3[1]}},
        'F': {'activation': tanh, 'edges': {}},
    }
    input_values = {
                    'A': inputs_vectors[0],
                    'B': inputs_vectors[1],
                    'C': inputs_vectors[2]
    }
    
    graph = SphericalEngine(adjacency_dict=adjacency_matrix, max_steps=6)
    result = graph.inference(verbose=True, input_values=input_values)
    output_nodes = result["output_nodes"]
    energy = result["energy_used"]
    assert energy == 7./6.
    output_value = output_nodes[0].state
    
    node_d_activation = adjacency_matrix.get('D', {}).get('activation', callable)
    node_e_activation = adjacency_matrix.get('E', {}).get('activation', callable)
    node_f_activation = adjacency_matrix.get('F', {}).get('activation', callable)
    
    layer_d = node_d_activation(np.dot(inputs_vectors, weights_vector1))
    layer_e = node_e_activation(np.dot(inputs_vectors[1:], weights_vector2))
    layer_ed_vector = np.array([layer_d, layer_e])
    
    assert abs(output_value - node_f_activation(np.dot(layer_ed_vector, weights_vector3))) < TOL
    

def test_behavior_4():
    weight_ab = np.random.randn()
    weight_ac = np.random.randn()
    weight_bc = np.random.randn()
    weight_bd = np.random.randn()
    weight_cd = np.random.randn()
    adjacency_matrix = {
        'A': {'activation': sigmoid, 'edges': {
                                                'B': weight_ab,
                                                'C': weight_ac
                                                }},
        'B': {'activation': relu, 'edges': {
                                                'C': weight_bc,
                                                'D': weight_bd
                                                }},
        'C': {'activation': tanh, 'edges': {'D': weight_cd}},
        'D': {'activation': identity, 'edges': {}}
    }
    input_values = {'A': np.random.randn()} 
    graph = SphericalEngine(adjacency_dict=adjacency_matrix, max_steps=6)
    result = graph.inference(verbose=True, input_values=input_values)
    output_nodes = result["output_nodes"]
    energy = result["energy_used"]
    
    assert energy == 6./4.
    
    node_a_activation = adjacency_matrix.get('A', {}).get('activation', callable)
    node_b_activation = adjacency_matrix.get('B', {}).get('activation', callable)
    node_c_activation = adjacency_matrix.get('C', {}).get('activation', callable)
    # Step 1
    b = node_b_activation(node_a_activation(input_values.get('A', 0.)) * weight_ab)
    c = node_c_activation(node_a_activation(input_values.get('A', 0.)) * weight_ac)
    # Step 2
    d = b * weight_bd + c * weight_cd
    c = node_c_activation(c + b * weight_bc)
    # Step 3
    d = d + c * weight_cd
    output_value = output_nodes[0].state
    assert abs(d - output_value) < TOL


def test_behavior_5():
    weight_ab = np.random.randn()
    adjacency_matrix = {
        'A': {'activation': identity, 'edges': {'B': weight_ab}},
        'B': {'activation': identity, 'edges': {}}
    }
    input_values = {'A': np.random.randn()}
    graph = SphericalEngine(adjacency_dict=adjacency_matrix, max_steps=6)
    result = graph.inference(verbose=True, input_values=input_values)
    output_nodes = result["output_nodes"]
    energy = result["energy_used"]
    
    assert energy == 1. / 2.
    
    input_val = input_values.get('A', 0.)
    node_a_activation = adjacency_matrix.get('A', {}).get('activation', callable)
    node_b_activation = adjacency_matrix.get('B', {}).get('activation', callable)
    state_a = node_a_activation(input_val)
    state_b = node_b_activation(state_a * weight_ab) 
    assert output_nodes[0].state == state_b
    
    # Traverse again without resetting
    result = graph.inference(verbose=True, input_values=input_values)
    output_nodes = result["output_nodes"]
    energy = result["energy_used"]
    state_a = node_a_activation(state_a + input_val)
    state_b = node_b_activation(state_a * weight_ab + state_b)
    assert output_nodes[0].state == state_b
    
    # Traverse again after resetting    
    graph.reset()
    result = graph.inference(verbose=True, input_values=input_values)
    output_nodes = result["output_nodes"]
    energy = result["energy_used"]
    state_a = node_a_activation(input_val)
    state_b = node_b_activation(state_a * weight_ab) 
    assert output_nodes[0].state == state_b
    
    
def test_behavior_6():
    weight_ab = np.random.randn()
    weight_bc = np.random.randn()
    weight_ca = np.random.randn()
    adjacency_matrix = {
        'A':  {'activation': sigmoid, 'edges': {'B': weight_ab}},
        'B':  {'activation': sigmoid, 'edges': {'C': weight_bc}},
        'C':  {'activation': sigmoid, 'edges': {'A': weight_ca}},
    }
    input_values = {
        'A': np.random.rand(),
        'B': np.random.rand(),
        'C': np.random.rand(),
    }
    graph = SphericalEngine(
        adjacency_dict=adjacency_matrix,
        max_steps=1)
    result = graph.inference(verbose=True, input_values=input_values)
    energy = result["energy_used"]
    assert energy == 1.
    
    graph = SphericalEngine(
        adjacency_dict=adjacency_matrix,
        max_steps=6)
    result = graph.inference(verbose=True, input_values=input_values)
    energy = result["energy_used"]
    assert energy == 6.
    
    NUM_STEPS:int = 6
    graph = SphericalEngine(
        adjacency_dict=adjacency_matrix,
        max_steps=NUM_STEPS)
    graph.inference(input_values=input_values)
    
    input_a = input_values.get('A', 0.0) 
    input_b = input_values.get('B', 0.0) 
    input_c = input_values.get('C', 0.0)
    node_a_activation = adjacency_matrix.get('A', {}).get('activation', callable)
    node_b_activation = adjacency_matrix.get('B', {}).get('activation', callable)
    node_c_activation = adjacency_matrix.get('C', {}).get('activation', callable)
    
    next_step_a = next_step_b = next_step_c = 0
    prev_step_a = node_a_activation(input_a)  
    prev_step_b = node_b_activation(input_b)  
    prev_step_c = node_c_activation(input_c)  

    for _ in range(NUM_STEPS):
        # Think of the signal from node X to node Y to need some time to propagate
        next_step_a:float = node_a_activation(prev_step_c * weight_ca + prev_step_a)
        next_step_b:float = node_b_activation(prev_step_a * weight_ab+ prev_step_b)
        next_step_c:float = node_c_activation(prev_step_b * weight_bc + prev_step_c)
        prev_step_a = next_step_a
        prev_step_b = next_step_b
        prev_step_c = next_step_c
        
    node_a:Node = cast(Node, graph._nodes.get('A', Node))
    node_b:Node = cast(Node, graph._nodes.get('B', Node))
    node_c:Node = cast(Node, graph._nodes.get('C', Node))
    assert abs(node_a.state - next_step_a) < TOL
    assert abs(node_b.state - next_step_b) < TOL
    assert abs(node_c.state - next_step_c) < TOL