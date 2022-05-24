import torch
from copy import deepcopy


# normalize
def normalize(original_graph, norm_dict):
    graph = deepcopy(original_graph)

    graph.ground_motions = graph.ground_motions / norm_dict['ground_motion']

    graph.x[:, :3] = graph.x[:, :3] / norm_dict['grid_num']
    graph.x[:, 3:6] = graph.x[:, 3:6] / norm_dict['coord']
    graph.x[:, 6] = graph.x[:, 6] / norm_dict['period']
    graph.x[:, 9] = graph.x[:, 9] / norm_dict['mass']
    graph.x[:, 10] = graph.x[:, 10] / norm_dict['node_inertia']
    graph.x[:, 11:14] = graph.x[:, 11:14] / norm_dict['modal_shape']
   
    graph.x[:, list(range(14, 26, 2))] = graph.x[:, list(range(14, 26, 2))] / norm_dict['elem_length']
    graph.x[:, list(range(15, 26, 2))] = graph.x[:, list(range(15, 26, 2))] / norm_dict['momentZ']

    assert graph.x.shape[1] == 36

    return graph





# denormalize
def denormalize_ground_motion(original_gm, norm_dict):
    gm = deepcopy(original_gm)
    gm = gm * norm_dict['ground_motion'] 
    return gm

def denormalize_grid_num(original_x, norm_dict):
    x = deepcopy(original_x)
    x = x * norm_dict['grid_num']
    return x

def denormalize_x(original_x, norm_dict):
    x = deepcopy(original_x)
    x[:, :3] = x[:, :3] * norm_dict['grid_num']
    x[:, 3:6] = x[:, 3:6] * norm_dict['coord']
    return x

def denormalize_acc(original_acc, norm_dict):
    acc = deepcopy(original_acc)
    acc = acc * norm_dict['acc']
    return acc

def denormalize_vel(original_vel, norm_dict):
    vel = deepcopy(original_vel)
    vel = vel * norm_dict['vel']
    return vel

def denormalize_disp(original_disp, norm_dict):
    disp = deepcopy(original_disp)
    disp = disp * norm_dict['disp']
    return disp

def denormalize_Mz(original_Mz, norm_dict):
    Mz = deepcopy(original_Mz)
    Mz = Mz * norm_dict['momentZ']
    return Mz

def denormalize_Sy(original_Sy, norm_dict):
    Sy = deepcopy(original_Sy)
    Sy = Sy * norm_dict['shearY']
    return Sy

