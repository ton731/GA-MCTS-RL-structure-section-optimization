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
simulator_path = "E:/TimeHistoryAnalysis/Results/Nonlinear_Dynamic_Analysis_Random/2022_04_27__15_39_08/"
designer = StructureDesigner.StructureDesigner(simulator_path=simulator_path)



# 3. Use MCTS to get the best beam_column design list.
mcts = MonteCarloTreeSearch.MCTS(designer=designer)
final_design = mcts.take_action(20)
designer.output_design(final_design)
print()
print("Final design:")
print(final_design, end='\n\n')



