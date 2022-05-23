import torch
import random
import numpy as np
import json


from Models import LSTM
from Utils import geometry
from Utils import normalization
from SearchMethod import MonteCarloTreeSearch



# Settings
# random seed
SEED = 731
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

# device
device = "cuda:0" if torch.cuda.is_available() else "cpu"
GPU_name = torch.cuda.get_device_name()
print("My GPU is {}\n".format(GPU_name))

# paths
save_model_dir = "E:/TimeHistoryAnalysis/Results/Nonlinear_Dynamic_Analysis_Random/2022_04_27__15_39_08/"
args_path = save_model_dir + "training_args.json"
norm_dict_path = save_model_dir + "norm_dict.json"
args = json.load(open(args_path, 'r'))
norm_dict = json.load(open(norm_dict_path, 'r'))

ipt_path = "Files/Input/structure.ipt"
gm_path = "Files/Input/ground_motion.txt"
candidate_path = "Files/Candidate/candidate.ipt"
analysis_path = "Files/Analysis/modal.ipt"

yield_factor = 0.2


# 1. First load in the .ipt file which the structure you want to design,
#    then construct it as a graph, maks element_category_list.
#    In element_category_list, the index order should come from 1F column, 2F column, ...., 1F beam, ..., NF beam.
graph, geo_info = geometry.get_graph_and_index_from_ipt(ipt_path, gm_path)
element_node_dict, node_element_dict, element_category_list, element_length_list = geo_info
graph = normalization.normalize(graph, norm_dict, yield_factor)
# print(f"max ground motion: {torch.max(graph.ground_motion)}")
# print(graph.ground_motion)


My_dict = geometry.get_norm_My_dict(norm_dict)
area_dict = geometry.get_section_area_dict(norm_dict)
min_max_usage = geometry.get_min_max_usage(element_category_list, element_length_list, area_dict)
total_elements = len(element_category_list)
print("Graph:", graph, end='\n\n')




# 2. Load in the LSTM model.
model_constructor_args = {
    'input_dim': graph.x.shape[1], 'hidden_dim': args["hidden_dim"], 'output_dim': 15,
    'num_layers': args["num_layers"]}
model = LSTM.LSTM(**model_constructor_args).to(device)
# print(model)




# 3. Use MCTS to get the best beam_column design list.
mcts_args = {"total_elements": total_elements, "element_category_list": element_category_list,
             "element_node_dict": element_node_dict, "node_element_dict": node_element_dict, "graph": graph, 
             "candidate_path":candidate_path, "analysis_path": analysis_path, "ipt_path": ipt_path, 
             "norm_dict": norm_dict, "model": model, "area_dict": area_dict, "element_length_list": element_length_list,
             "min_max_usage": min_max_usage, "My_dict": My_dict, "yield_factor": yield_factor}
mcts = MonteCarloTreeSearch.MCTS(mcts_args)
final_design = mcts.take_action(5000)
print()
print("Final design:")
print(final_design, end='\n\n')





# 4. Use the best design list to reconstruct .ipt file 
output_path = "Files/Output/design.ipt"
geometry.reconstruct_ipt_file(ipt_path, output_path, final_design, node_element_dict)
