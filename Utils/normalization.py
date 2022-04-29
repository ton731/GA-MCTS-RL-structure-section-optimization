import torch
from copy import deepcopy



def normalize(original_graph, norm_dict):
    graph = deepcopy(original_graph)

    graph.ground_motion = graph.ground_motion / norm_dict['ground_motion']

    graph.x[:, :3] = graph.x[:, :3] / norm_dict['grid_num']
    graph.x[:, 3:6] = graph.x[:, 3:6] / norm_dict['coord']
    graph.x[:, 6] = graph.x[:, 6] / norm_dict['period']
    graph.x[:, 9] = graph.x[:, 9] / norm_dict['mass']
    graph.x[:, 10] = graph.x[:, 10] / norm_dict['node_inertia']
    graph.x[:, 11:14] = graph.x[:, 11:14] / norm_dict['modal_shape']
   
    graph.x[:, 14:20] = graph.x[:, 14:20] / norm_dict['elem_length']
    graph.x[:, 20:26] = graph.x[:, 20:26] / norm_dict['momentZ']

    assert graph.x.shape[1] == 36

    return graph