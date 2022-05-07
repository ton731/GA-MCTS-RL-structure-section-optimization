import torch
from copy import deepcopy


My_start_index = 20
def reset_plastic_threshold(graph, yield_factor=0.85):
    for node_index in range(graph.x.shape[0]):
        for i, face_index in enumerate(list(range(3, 9))):   # face_index is index for Mz(x_n, x_p, y_n, y_p, z_n, z_p)
            real_moment_face_i = graph.y[node_index, :, face_index]     # [2000]
            My_localZ_face_i = graph.x[node_index, My_start_index + i]
            if My_localZ_face_i <= 0.1:     # It means this face is not connect to any element
                continue
            real_node_plastic_hinge = (real_moment_face_i >= yield_factor * My_localZ_face_i) + 0  # [2000]

            # Once plastic hinge occurs, make the later rest all 1
            if(torch.max(real_node_plastic_hinge) != 0):
                for timestep in range(graph.y.shape[1]):
                    if(real_node_plastic_hinge[timestep] == 1):
                        graph.y[node_index, timestep:, face_index] = 1
                        break
    return graph



# normalize + reset My threshold
def normalize(original_graph, norm_dict, yield_factor):
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

    graph = reset_plastic_threshold(graph, yield_factor)

    return graph