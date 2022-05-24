import torch
import random
import numpy as np
import sys

sys.path.append('Utils/')
sys.path.append('Searching/')
sys.path.append('Models/')


from Models import LSTM
from Utils import geometry
from Utils import normalization
from Searching import MonteCarloTreeSearch
import StructureDesigner



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



# Structure Designer
simulator_path = "E:/TimeHistoryAnalysis/Results/Nonlinear_Dynamic_Analysis_ChiChi_Taipei3_Real/2022_05_22__14_11_10"
ground_motion_number = 3
designer_args = {"simulator_path": simulator_path, "ground_motion_number": ground_motion_number,
                 "mode": "story", "method": "MCTS"}
designer = StructureDesigner.StructureDesigner(**designer_args)



# 3. Use MCTS to get the best beam_column design list.
rounds = 2
mcts = MonteCarloTreeSearch.MCTS(designer=designer)
final_design = mcts.take_action(rounds)
designer.output_design(final_design)
print()
print("Final design:")
print(final_design, end='\n\n')
designer.visualize_response(final_design)



