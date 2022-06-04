import torch
import json
import os
import random
import numpy as np
from datetime import datetime
from os.path import join
from copy import deepcopy

from Utils import geometry
from Utils import normalization
from Models import LSTM
from Utils import analysis
from Utils import evaluation
from Utils import visualization




class StructureSimulator:
    def __init__(self, simulator_path, ground_motion_number, method='MCTS', comment=None):

        # 1. Simulator Enviroment
        self.method = method
        self.comment = comment
        self.time = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.simulator_path = simulator_path
        self.model_path = join(self.simulator_path, "model.pt")
        self.args_path = None
        self.norm_dict_path = None
        self.args = None
        self.norm_dict = None
        self.yield_factor = None
        self.My_dict = None
        self.area_dict = None
        
        
        # 2. Simulator processing folder
        self.simulation_processing_dir = "Files"
        self.gm_folder = None
        self.analysis_path = None
        self.output_folder = None
        self.output_path = None
        self.record_path = None


        # 3. Simulator Model
        self.model_constructor_args = None
        self.model = None


        # 4. Design-level ground motions initialization
        self.ground_motion_number = ground_motion_number
        self.ground_motions = None
        self.target_objectives = None
        self.gm_names = None


        # 4. Initialization
        self._initialize_simulator_envrimoment()
        self._initialize_simulator_processing_folder()
        self._initialize_simulator_model()
        self._initialize_design_groundMotions()


    def __str__(self):
        return "Structure Analysis Simulator based on StrucLSTM model."


    def _initialize_simulator_envrimoment(self):
        print("Initializing AI-Based structural analysis simulator......")
        # random seed
        SEED = 731
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed(SEED)
        torch.backends.cudnn.deterministic = True
        self.args_path = join(self.simulator_path, "training_args.json")
        self.norm_dict_path = join(self.simulator_path, "norm_dict.json")
        self.args = json.load(open(self.args_path, 'r'))
        self.norm_dict = json.load(open(self.norm_dict_path, 'r'))
        self.yield_factor = self.args['yield_factor']
        self.My_dict = geometry.get_norm_My_dict(self.norm_dict)
        self.area_dict = geometry.get_section_area_dict(self.norm_dict)


    def _initialize_simulator_processing_folder(self):
        self.gm_folder = join(self.simulation_processing_dir, "Input/GroundMotions")
        self.analysis_path = join(self.simulation_processing_dir, "Analysis/modal.ipt")

        self.output_folder = join(self.simulation_processing_dir, f"Output/{self.method}")
        if os.path.exists(self.output_folder) == False:
            os.mkdir(self.output_folder)
        self.output_folder = join(self.output_folder, self.time)
        if os.path.exists(self.output_folder) == False:
            os.mkdir(self.output_folder)
        self.output_path = join(self.output_folder, "design.ipt")

        with open(join(self.output_folder, "comment.txt"), "w") as f:
            f.write(self.comment)

        self.record_path = join(self.output_folder, "record.txt")


    def _initialize_simulator_model(self):
        self.model_constructor_args = {
            'input_dim': 36, 'hidden_dim': self.args["hidden_dim"], 'output_dim': 15,
            'num_layers': self.args["num_layers"]}
        self.model = LSTM.LSTM(**self.model_constructor_args).to(self.device)
        self.model.load_state_dict(torch.load(self.model_path))
        print("LSTM model: ")
        print(self.model)


    def _initialize_design_groundMotions(self):
         data = geometry.get_design_groundMotions_from_folder(self.gm_folder, self.ground_motion_number)
         self.ground_motions, self.target_objectives, self.gm_names = data
         self.ground_motions = normalization.normalize_ground_motion(self.ground_motions, self.norm_dict)
         self.ground_motions = self.ground_motions.to(self.device)


    def score(self, designer, candidate_design):
        # First, if mode is 'story', need to convert story design to element design
        candidate_design = designer.get_design(candidate_design)
        modal_result = analysis.run_modal_analysis(designer, self, candidate_design)
        candidate_graph = analysis.make_section_graph(designer, self, designer.graph, candidate_design, modal_result)
        response = analysis.predict(self, candidate_graph)
        score = evaluation.designScore(designer, self, candidate_graph, response, candidate_design)
        return score

    
    def batch_score(self, designer, batch_design, batch_size):
        candidate_designs = [designer.get_design(design) for design in batch_design]
        modal_results = [analysis.run_modal_analysis(designer, self, candidate_design) for candidate_design in candidate_designs]
        candidate_graphs = [analysis.make_section_graph(designer, self, designer.graph, candidate_design, modal_result) for candidate_design, modal_result in zip(candidate_designs, modal_results)]
        responses = analysis.batch_predict(self, candidate_graphs, batch_size)
        scores = evaluation.batch_designScore(designer, self, candidate_graphs, responses, candidate_designs, batch_size)
        return scores


    def visualize_response(self, designer, final_design):
        response_folder = join(self.output_folder, "Response")
        os.mkdir(response_folder)

        modal_result = analysis.run_modal_analysis(designer, self, final_design)
        final_graph = analysis.make_section_graph(designer, self, designer.graph, final_design, modal_result)
        response = analysis.predict(self, final_graph)  # [node_num * gm_num, 2000, 15]

        node_num = designer.graph.x.shape[0]
        for gm_index in range(self.ground_motion_number):
            target_level = self.target_objectives[gm_index]
            print(f"Visualizing response corresponding with ground motion {gm_index+1} ({target_level})...")
            individual_folder = join(response_folder, f"GroundMotion_{gm_index+1}_{target_level}")
            os.mkdir(individual_folder)
            ground_motion = self.ground_motions[:, gm_index*10 : gm_index*10 + 10]     # [2000, 10]
            output = response[node_num*gm_index:node_num*(gm_index+1), :, :]
            visualization.visualize_ground_motion(individual_folder, ground_motion, self.norm_dict, self.gm_names[gm_index])
            for response_item in ["Acceleration", "Velocity", "Displacement", "Moment_Z_Column", "Moment_Z_Xbeam", "Shear_Y"]:
                visualization.visualize_response(individual_folder, designer.graph.x, output, self.norm_dict, response_item)
            visualization.visualize_plasticHinge(individual_folder, designer.graph.x, output, self.norm_dict)


