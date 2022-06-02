import torch
import numpy as np
import normalization


def designScore(designer, simulator, graph, response, design):
    '''
    Score will be contributed by PlasticHingeScore and MaterialUsageScore
    '''
    materialFactor, strengthFactor = 0.4, 0.6
    score = 0

    materialScore = materialUsageScore(designer, simulator, design)
    strengthScore = plasticHingeScore(designer, simulator, graph, response)
    # score += driftRatioScore(designer, simulator, graph, response)

    score += materialFactor * materialScore + strengthFactor * strengthScore
    
    designer.weighted_score.append(score)
    designer.material_score.append(materialScore)
    designer.strength_score.append(strengthScore)

    return score






def materialUsageScore(designer, simulator, candidate_design):
    min_usage, max_usage = designer.min_max_usage

    # get design's total material usage
    total_usage = 0
    for i, section in enumerate(candidate_design):
        total_usage += designer.element_length_list[i] * simulator.area_dict[section]
    
    # normalize usage with given min, max usage
    total_usage = (total_usage - min_usage) / (max_usage - min_usage)   # 0 ~ 1, 0 is less material usage
    score = 1 - total_usage
    # score = total_usage

    return score




# Life safety        : transient 2.5%, permanent 1%
# Collapse prevention: transient 5%,   permanent 5%
# Only see the displacement of [0, y, 0] my cause some problem due to the inaccuracy prediction on those point.
# Instead, since the displacements in the same story should be the same,
# we should calculate the mean of the displacements of the nodes in the same story.
# If the mean displacement exceed the tolerance of displacement in the FEMA273 code, we then consider it unsafe.
# To elliviate some inaccuracy prediction, we only take those middle values of the displacements.
LS_transient, LS_permanent = 0.025, 0.01
CP_transient, CP_permanent = 0.05, 0.05
def driftRatioScore(designer, simulator, graph, response):
    score = simulator.ground_motion_number  # Every ground motion can get 1 point at first, if exceed drift ratio then minus the point.
    node_num = graph.x.shape[0]
    for gm_index in range(simulator.ground_motion_number):
        gm_response = response[node_num*gm_index:node_num*(gm_index+1), :]
        # print("GM_{gm_index}, In drift ratio evaluation, gm_response shape: ", gm_response.shape)
        target_level = simulator.target_objectives[gm_index]
        exceed_regulation_num = 0
        for story_name in designer.node_drift_node_dict.keys():
            story_node_num = len(designer.node_drift_node_dict[story_name])
            story_drift_norm = []
            for node_pair in designer.node_drift_node_dict[story_name]:
                story, story_height, node_index_top, node_index_bottom = node_pair
                drift_norm = torch.max(abs(gm_response[node_index_top, :, 0] - gm_response[node_index_bottom, :, 0]))
                story_drift_norm.append(drift_norm.cpu().numpy())

            story_drift_norm = np.array(story_drift_norm)
            story_drift_norm = np.sort(story_drift_norm)
            assert story_node_num == len(story_drift_norm)
            story_drift_norm_mean = np.mean(story_drift_norm[int(story_node_num*1/3):-int(story_node_num*1/3)])
            story_drift_mean = normalization.denormalize_disp(story_drift_norm_mean, simulator.norm_dict)
            story_drift_ratio_mean = story_drift_mean / story_height

            transient_tolerance = LS_transient if target_level == "BSE-1" else CP_transient
            print(f"{story_name} story drift ratio: {story_drift_ratio_mean}, tolerance: {transient_tolerance}")
            if story_drift_ratio_mean > transient_tolerance:
                exceed_regulation_num += 1

        print(f"In {designer.story_num}F structure, there are {exceed_regulation_num}F exceed code regulated drift ratio.")
        if exceed_regulation_num > 1:
            score -= 1
    
    score /= simulator.ground_motion_number
    return score





My_start_index = 15
section_info_dim = 2
yield_factor = 0.95
allowable_plastic_hinge_number_per_section = 20     # 20 hinge score = 1, 100 hinge score = 0
tolerant_plastic_hinge_number_per_section = 100 
def plasticHingeScore(designer, simulator, graph, response):
    # response: [node_num, 2000(timestep), 15(out_dim)]
    x = graph.x
    node_num = graph.x.shape[0]
    allowable_plastic_hinge_number = allowable_plastic_hinge_number_per_section * simulator.ground_motion_number
    tolerant_plastic_hinge_number = tolerant_plastic_hinge_number_per_section * simulator.ground_motion_number
    total_plastic_hinge_num = 0
    for gm_index in range(simulator.ground_motion_number):
        gm_response = response[node_num*gm_index:node_num*(gm_index+1), :]
        plastic_hinge_shape = [gm_response.shape[0], 6]    # [node_num, 6 face]
        ph_pred = torch.zeros(plastic_hinge_shape)

        for node_index in range(node_num):
            for i, face_index in enumerate(list(range(3, 9))):   # face_index is index for Mz(x_n, x_p, y_n, y_p, z_n, z_p)
                pred_moment_face_i = gm_response[node_index, :, face_index]     # [2000]
                My_localZ_face_i = x[node_index, My_start_index + i * section_info_dim]
                if My_localZ_face_i <= 0.1:     # It means this face is not connect to any element
                    continue
                pred_node_plastic_hinge = (abs(pred_moment_face_i) >= yield_factor * My_localZ_face_i) + 0  # [2000]

                # Once plastic hinge occurs, set the [index, face] as plastic
                if(torch.max(pred_node_plastic_hinge) != 0):
                    # print('There is a plastic hinge!')
                    ph_pred[node_index, i] = 1

        plastic_num = torch.sum(ph_pred)
        total_plastic_hinge_num += plastic_num
        # print(f"plastic hinge number: {plastic_num}")

    # Maximum tolerable plastic hinge number is all 1F column yield --> 1F column number * 2 (bottom and top of a column)
    # print(f"plastic hinge number: {total_plastic_hinge_num }")
    score = 1 - (total_plastic_hinge_num - allowable_plastic_hinge_number) / (tolerant_plastic_hinge_number - allowable_plastic_hinge_number)
    score = max(torch.tensor(0), score)   # In case score < 0
    score = min(torch.tensor(1), score)   # In case score > 1
    score = score.cpu().numpy()

    return score


