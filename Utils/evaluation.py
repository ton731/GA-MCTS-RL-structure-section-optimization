import torch


def designScore(designer, simulator, graph, response, design):
    '''
    Score will be contributed by PlasticHingeScore and MaterialUsageScore
    '''
    strengthFactor, materialFactor = 0.7, 1.0
    score = 0
    # score += strengthFactor * plasticHingeScore(designer, simulator, graph, response)
    score += materialFactor * materialUsageScore(designer, simulator, design)
    return score



def driftRatioScore(designer, simulator, graph, response):
    pass



def materialUsageScore(designer, simulator, candidate_design):
    min_usage, max_usage = designer.min_max_usage

    # get design's total material usage
    total_usage = 0
    for i, section in enumerate(candidate_design):
        total_usage += designer.element_length_list[i] * simulator.area_dict[section]
    
    # normalize usage with given min, max usage
    total_usage = (total_usage - min_usage) / (max_usage - min_usage)   # 0 ~ 1, 0 is less material usage
    # score = 1 - total_usage
    score = total_usage

    return score




'''
My_start_index = 15
section_info_dim = 2
def plasticHingeScore(designer, simulator, graph, response):
    # response: [node_num, 2000(timestep), 15(out_dim)]
    x = graph.x
    plastic_hinge_shape = [response.shape[0], 6]    # [node_num, 6 face]
    ph_pred = torch.zeros(plastic_hinge_shape)

    for node_index in range(ph_pred.shape[0]):
        for i, face_index in enumerate(list(range(3, 9))):   # face_index is index for Mz(x_n, x_p, y_n, y_p, z_n, z_p)
            pred_moment_face_i = response[node_index, :, face_index]     # [2000]
            My_localZ_face_i = x[node_index, My_start_index + i * section_info_dim]
            if My_localZ_face_i <= 0.1:     # It means this face is not connect to any element
                continue
            pred_node_plastic_hinge = (abs(pred_moment_face_i) >= simulator.yield_factor * My_localZ_face_i) + 0  # [2000]

            # Once plastic hinge occurs, set the [index, face] as plastic
            if(torch.max(pred_node_plastic_hinge) != 0):
                # print('There is a plastic hinge!')
                ph_pred[node_index, i] = 1

    plastic_num = torch.sum(ph_pred)

    # Maximum tolerable plastic hinge number is all 1F column yield --> 1F column number * 2 (bottom and top of a column)
    x_grid_num, y_grid_num, z_grid_num = graph.grid_num
    maximum_tolerable_plastic_hinge_number = 2 * (x_grid_num * z_grid_num)
    score = 1 - plastic_num / maximum_tolerable_plastic_hinge_number
    # print(f"plastic hinge num: {plastic_num}")
    score = score.cpu().numpy()

    return score
'''

