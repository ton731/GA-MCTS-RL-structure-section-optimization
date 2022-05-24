import os
import torch
import time


pisa = "E:/StructureInverseDesign/Files/Analysis/PISA3D_Batch_500nodes.exe"
eigen_file_path = "E:/StructureInverseDesign/Files/Analysis/MODAL.Eigen"
# modal_file_path = "E:/StructureInverseDesign/Files/Analysis/MODAL.Modal"


def run_modal_analysis(designer, candidate_design):
    input_file = open(designer.ipt_path, 'r').readlines()
    with open(designer.analysis_path, 'w') as f:
        for line in input_file:
            if "# Analysis  ModeShape" in line:
                new_line = line.replace("# Analysis  ModeShape", "Analysis  ModeShape")
                f.write(new_line)
            elif "Analysis  Dynamic" in line:
                new_line = line.replace("Analysis  Dynamic", "# Analysis  Dynamic")
                f.write(new_line)
            elif "Element  BeamColumn" in line:
                contents = line.split()
                node1_name, node2_name = contents[3], contents[4]
                element_index = designer.node_element_dict[f"{node1_name}_{node2_name}"]
                contents[5] = candidate_design[element_index]
                new_line = " ".join(contents)
                f.write(new_line + '\n')
            elif "LoadPattern  GroundAccel" in line:
                new_line = line.replace("LoadPattern  GroundAccel", "# LoadPattern  GroundAccel")
                f.write(new_line)
            else:
                f.write(line)

    # Run modal analysis
    modal_path = designer.analysis_path.split(".")[0]
    # Hide os.system output from terminal:
    # https://stackoverflow.com/questions/33985863/hiding-console-output-produced-by-os-system
    # os.system(pisa + " " + modal_path + " " + ">/null 2>&1")
    finished = False
    while not finished:
        os.system(pisa + " " + modal_path + " " + ">/null 2>&1")
        finished = check_modal_analysis()
    time.sleep(0.1)


    # Get period and first mode shape from modal result
    # eigen_file = open(eigen_file_path, 'r').readlines()
    first_mode_period = None
    is_mode_1 = False
    node_first_mode_shape = torch.zeros((designer.graph.x.shape[0], 3))

    with open(eigen_file_path, 'r') as f:
        for line in f.readlines():
            if "Period of Mode 1" in line:
                contents = line.strip().split()
                first_mode_period = float(contents[5])
            elif "Mode 1, Period =" in line:
                is_mode_1 = True
            elif "----------" in line:
                is_mode_1 = False
            elif is_mode_1 == True:
                if "Node" in line or "N" not in line:   continue
                contents = line.strip().split()
                node_index = int(contents[0][1:]) - 1
                node_first_mode_shape[node_index, 0] = float(contents[1])
                node_first_mode_shape[node_index, 1] = float(contents[3])
                node_first_mode_shape[node_index, 2] = float(contents[5])
    
    return (first_mode_period, node_first_mode_shape)


def check_modal_analysis():
    if os.path.exists("Files/Analysis/MODAL.Eigen"):
        return True
    return False




def make_section_graph(designer, graph, candidate_design, modal_result):
    # element_node_dict --> key: element_index, value: [node1_index, node2_index, node1_face (My's index), node2_face]
    for element_index, section in enumerate(candidate_design):
        node1_index, node2_index, node1_face, node2_face = designer.element_node_dict[element_index]
        graph.x[node1_index, node1_face] = designer.My_dict[section]
        graph.x[node2_index, node2_face] = designer.My_dict[section]
    
    # modal result: normalize --> add to node feature
    first_mode_period, node_first_mode_shape = modal_result
    first_mode_period /= designer.norm_dict["period"]
    node_first_mode_shape /= designer.norm_dict["modal_shape"] 
    graph.x[:, 6] = first_mode_period
    graph.x[:, 11:14] = node_first_mode_shape

    return graph




def predict(designer, candidate_graph):
    graph = candidate_graph.to(designer.device)
    duplicate_x = graph.x.repeat(designer.ground_motion_number, 1)    # duplicate x from [graph_nodes, features] to [graph_nodes * gm_num, features]

    designer.model.eval()
    with torch.no_grad():
        output = torch.zeros((graph.y.shape[0] * designer.ground_motion_number, graph.y.shape[1], designer.model.output_dim)).to(designer.device)
        H_list = [None for i in range(designer.model.num_layers)]
        C_list = [None for i in range(designer.model.num_layers)]
        for i, gm in enumerate(graph.ground_motions):
            H_list, C_list, out = designer.model(gm, duplicate_x, None, None, graph.ptr, H_list, C_list)
            output[:, i, :] = out
    return output

