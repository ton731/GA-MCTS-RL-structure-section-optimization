import torch
from torch_geometric.data import Data
import numpy as np
import eqsig.single
from .sections import *



def scale_gm_to_design_level(gm_path, original_gm):
    min_t = 0.45
    max_t = 1.35
    main_start, main_end = None, None    # indexes for t = 0.45 ~ 1.35
    main_Sa = None                       # mean for Sa between t = 0.45 ~ 1.35

    # 2. Calculate design spectrum 
    S1D, SSD = 0.45, 0.80
    Bs, B1 = 0.8, 0.8
    Na, Nv = 1.37, 1.44
    Fa, Fv = 1.0, 1.2
    S1 = S1D * Nv
    Ss = SSD * Na
    SX1 = Fv * S1
    SXS = Fa * Ss
    T0 = (SX1 * Bs) / (SXS * B1)
    print(f"T0 = {T0:.5f}")

    periods = np.linspace(0.2, 10, 100)
    design_spectrum = np.zeros(1000)
    for i, t in enumerate(periods):
        if t < 0.2*T0:
            design_spectrum[i] = Fa * Ss / Bs * (0.4 + 3 * t / T0)
        elif t >= 0.2*T0 and t <= T0:
            design_spectrum[i] = Fa * Ss / Bs
        elif t > T0:
            design_spectrum[i] = Fv * S1 / (B1 * t)
        else:
            design_spectrum[i] = None

    for i, t in enumerate(periods):
        if main_start == None and t >= min_t:
            main_start = i
        if main_end == None and t > max_t:
            main_end = i

    main_Sa = np.sum(design_spectrum[main_start:main_end])


    dt = 0.005
    file = np.loadtxt(gm_path)
    gm = file[:, 1] / 1000 / 9.8
    record = eqsig.AccSignal(gm, dt)
    record.generate_response_spectrum(response_times=periods)

    sa_sum = np.sum(record.s_a[main_start:main_end])
    scale_factor = main_Sa / sa_sum

    scaled_gm = original_gm * scale_factor

    print(f"Ground motion is amplified {scale_factor:.4f} times to the design scale")

    return scaled_gm




def get_graph_and_index_from_ipt(ipt_path, gm_path):
    input_file = open(ipt_path, 'r').readlines()

    # Ground motion has to be amplified to code level.
    ground_motion = torch.zeros((2000, 10))
    with open(gm_path, "r") as f:
        for index, line in enumerate(f.readlines()):
            i, j = index//10, index%10
            ground_motion[i, j] = float(line.split()[1])

    ground_motion = scale_gm_to_design_level(gm_path, ground_motion)


    node_dict = {}
    index_node_dict = {}
    node_count = 0
    node_grid_dict = {}
    node_coord_dict = {}            # node name -->  node coord
    coord_node_dict = {}            # node coord --> node index
    node_dof_dict = {}
    node_mass_dict = {}      
    beamcolumn_node_dict = {}      
    x_grid, y_grid, z_grid = [], [], []
    x_grid_space, z_grid_space = 0, 0


    for line in input_file:
        contents = line.split()
        if len(contents) == 0: continue
        
        if contents[0] == 'GUI_GRID':
            if contents[1] == 'XDIR':
                x_grid = [int(num) for num in contents[2:]]
                x_grid_space = x_grid[1] - x_grid[0]
            elif contents[1] == 'YDIR':
                y_grid = [int(num) for num in contents[2:]]
            elif contents[1] == 'ZDIR':
                z_grid = [int(num) for num in contents[2:]]
                z_grid_space = z_grid[1] - z_grid[0]
                
        
        if contents[0] == 'Node' and contents[1][0] != 'M':
            node_dict[contents[1]] = node_count
            index_node_dict[node_count] = contents[1]
            node_coord_dict[contents[1]] = [float(contents[2])/1000, float(contents[3])/1000, float(contents[4])/1000]
            coord_node_dict[f"{contents[2]}_{contents[3]}_{contents[4]}"] = node_count
            node_grid_dict[contents[1]] = [x_grid.index(int(contents[2])), y_grid.index(int(contents[3])), z_grid.index(int(contents[4]))]
            node_count += 1


        elif contents[0] == 'DOF' and contents[1][0] != 'M':
            dof = [int(contents[2]), int(contents[3]), int(contents[4]), int(contents[5]), int(contents[6]), int(contents[7])]
            if dof == [-1, -1, -1, -1, -1, -1]:   
                node_dof_dict[contents[1]] = 1
                
                
        elif contents[0] == '#NodeMass':
            mass, Rx, Ry, Rz = float(contents[3]), float(contents[6]), float(contents[7]), float(contents[8])
            node_mass_dict[contents[2]] = [mass, Rx, Ry, Rz]
            
            
        elif contents[0] == 'Element':
            node1, node2 = node_dict[contents[3]], node_dict[contents[4]]
            beamcolumn_node_dict[contents[2]] = [contents[3], contents[4]]
                             

    grid_num = torch.tensor([len(x_grid), len(y_grid), len(z_grid)])
    ptr = [0, node_count]

    x = torch.zeros((node_count, 36))
    y = torch.zeros((node_count, 2000, 21))

    for line in input_file:
        contents = line.split()
        if len(contents) == 0: continue
        
        if contents[0] == 'Node' and contents[1][0] != 'M':
            node_index = node_dict[contents[1]]
            node_name = contents[1]
            
            # structure X, Y, Z grid nums
            x[node_index][0] = len(x_grid)
            x[node_index][1] = len(y_grid)
            x[node_index][2] = len(z_grid)
            
            
            # node coordinates
            x[node_index][3] = node_grid_dict[contents[1]][0]
            x[node_index][4] = node_grid_dict[contents[1]][1]
            x[node_index][5] = node_grid_dict[contents[1]][2]

            
            # natural period
            x[node_index][6] = 0


            # DOF
            if contents[1] not in node_dof_dict.keys():
                x[node_index][7] = 1    # free
                x[node_index][8] = 0    # fixed
            else:
                x[node_index][7] = 0    # free
                x[node_index][8] = 1    # fixed
                
                
            # Mass
            mass, Rx, Ry, Rz = node_mass_dict[node_name]
            x[node_index][9] = mass
            x[node_index][10] = Ry


            # 1st mode shape (Ux, Uz, Ry)
            x[node_index][11] = 0
            x[node_index][12] = 0
            x[node_index][13] = 0


            # Ground Motion
            x[node_index][26:36] = 0

    graph = Data(x=x, y=y, ground_motion=ground_motion, grid_num=grid_num, ptr=ptr)



    # Add node's each face element length to node feature
    for line in input_file:
        contents = line.split()
        if len(contents) == 0: continue
        if contents[0] != 'Element': continue

        node1_index, node2_index = node_dict[contents[3]], node_dict[contents[4]]
        node1_name, node2_name = contents[3], contents[4]

        x1, y1, z1 = node_coord_dict[node1_name]
        x2, y2, z2 = node_coord_dict[node2_name]

        # length
        # x_n, x_p, y_n, y_p, z_n, z_p
        #  14   15   16   17   18   19

        if x1 != x2:
            # It's a x beam
            length = x2 - x1
            x[node1_index][15] = length
            x[node2_index][14] = length
        elif y1 != y2:
            # It's a column
            length = y2 - y1
            x[node1_index][17] = length
            x[node2_index][16] = length
        elif z1 != z2:
            # It's a z beam
            length = z2 - z1
            x[node1_index][19] = length
            x[node2_index][18] = length


    # Make element selection order
    # 0F col --> 1F col --...--> NF col --> 1F x beam --> 1F z beam --> 2F x beam --> 2F z beam --...--> NF z beam

    # My
    # x_n, x_p, y_n, y_p, z_n, z_p
    #  20   21   22   23   24   25

    element_count = 0
    element_category_list = []  # [0, 0, 0, 1, 1, 1, ......] beam(1) or column(0)
    element_node_dict = {}      # key: element_index, value: [node1_index, node2_index, node1_face (My's index), node2_face]
    node_element_dict = {}      # key: node1Name_node2Name, value: element_index
    element_length_list = []    # [l1, l2, l3, .....]

    for y in y_grid[:-1]:
        for x in x_grid:
            for z in z_grid:
                coord1 = f"{x}_{y}_{z}"
                coord2_y = y+3200 if y > 0 else y+4200
                coord2 = f"{x}_{coord2_y}_{z}"
                node1_index = coord_node_dict[coord1]
                node2_index = coord_node_dict[coord2]
                element_node_dict[element_count] = [node1_index, node2_index, 23, 22]
                node_element_dict[f"{index_node_dict[node1_index]}_{index_node_dict[node2_index]}"] = element_count
                element_count += 1
                element_category_list.append(0)  # 1 is beam, 0 is column
                element_length_list.append((coord2_y - y)/1000)

    for y in y_grid[1:]:
        # x beam
        for z in z_grid:
            for x in x_grid[:-1]:
                coord1 = f"{x}_{y}_{z}"
                coord2 = f"{x+x_grid_space}_{y}_{z}"
                node1_index = coord_node_dict[coord1]
                node2_index = coord_node_dict[coord2]
                element_node_dict[element_count] = [node1_index, node2_index, 21, 20]
                node_element_dict[f"{index_node_dict[node1_index]}_{index_node_dict[node2_index]}"] = element_count
                element_count += 1
                element_category_list.append(1)
                element_length_list.append(x_grid_space/1000)

        # z beam
        for x in x_grid:
            for z in z_grid[:-1]:
                coord1 = f"{x}_{y}_{z}"
                coord2 = f"{x}_{y}_{z+z_grid_space}"
                node1_index = coord_node_dict[coord1]
                node2_index = coord_node_dict[coord2]
                element_node_dict[element_count] = [node1_index, node2_index, 25, 24]
                node_element_dict[f"{index_node_dict[node1_index]}_{index_node_dict[node2_index]}"] = element_count
                element_count += 1
                element_category_list.append(1)
                element_length_list.append(z_grid_space/1000)



    # Check element number is right
    assert len(beamcolumn_node_dict.keys()) == element_count

    return graph, (element_node_dict, node_element_dict, element_category_list, element_length_list)




def reconstruct_ipt_file(ipt_path, output_file, design, node_element_dict):
    input_file = open(ipt_path, 'r').readlines()

    with open(output_file, 'w') as f:
        for line in input_file:
            contents = line.split()
            if len(contents) == 0 or contents[0] != 'Element':
                f.write(line)
                continue

            node1_name, node2_name = contents[3], contents[4]
            element_index = node_element_dict[f"{node1_name}_{node2_name}"]
            contents[5] = design[element_index]
            new_line = " ".join(contents)
            f.write(new_line + '\n')

    print("Reconstruct ipt file successfully.", end='\n'*3)






def get_norm_My_dict(norm_dict):
    My_dict = {}
    for section in beam_sections.keys():
        My_dict[section] = beam_sections[section][12][1] / norm_dict['momentZ']

    for section in column_sections.keys():
        My_dict[section] = column_sections[section][12][1] / norm_dict['momentZ']

    return My_dict


def get_section_area_dict(norm_dict):
    area_dict = {}
    for section in beam_sections.keys():
        area_dict[section] = beam_sections[section][12][1] / norm_dict['momentZ']

    for section in column_sections.keys():
        area_dict[section] = column_sections[section][12][1] / norm_dict['momentZ']

    return area_dict


def get_min_max_usage(element_category_list, element_length_list, area_dict):
    min_usage, max_usage = 0, 0
    for i, element_isBeam in enumerate(element_category_list):
        if(element_isBeam):
            max_area = area_dict['W21x93']
            min_area = area_dict['W21x44']
        else:
            max_area = area_dict['16x16x0.875']
            min_area = area_dict['16x16x0.375']

        min_usage += element_length_list[i] * min_area
        max_usage += element_length_list[i] * max_area   
    
    return [min_usage, max_usage]