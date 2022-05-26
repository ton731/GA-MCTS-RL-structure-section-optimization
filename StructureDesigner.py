import torch
import json
import os
from datetime import datetime
from os.path import join
from copy import deepcopy

from Utils import geometry
from Utils import normalization
from Models import LSTM
from Utils import analysis
from Utils import evaluation
from Utils import visualization



available_beam_sections = ['W21x93', 'W21x83', 'W21x73', 'W21x68', 'W21x62', 'W21x57', 'W21x50', 'W21x48', 'W21x44']
available_column_sections = ['16x16x0.875', '16x16x0.75', '16x16x0.625', '16x16x0.5', '16x16x0.375']



class StructureDesigner:
    def __init__(self, simulator_path, ground_motion_number=10, mode='element', method='MCTS'):
        self.comment = "MCTS, round=20, story mode"

        # 1. Simulator Enviroment
        self.mode = mode    # element / story
        self.method = method
        self.time = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.simulator_path = simulator_path
        self.args_path = None
        self.norm_dict_path = None
        self.args = None
        self.norm_dict = None
        self.simulation_processing_dir = "Files"
        self.ipt_path = os.path.join(self.simulation_processing_dir, "Input/structure.ipt")
        self.gm_folder = join(self.simulation_processing_dir, "Input/GroundMotions")
        self.candidate_path = join(self.simulation_processing_dir, "Candidate/candidate.ipt")
        self.analysis_path = join(self.simulation_processing_dir, "Analysis/modal.ipt")

        self.output_folder = join(self.simulation_processing_dir, f"Output/{method}")
        if os.path.exists(self.output_folder) == False:
            os.mkdir(self.output_folder)
        self.output_folder = join(self.output_folder, self.time)
        if os.path.exists(self.output_folder) == False:
            os.mkdir(self.output_folder)
        self.output_path = join(self.output_folder, "design.ipt")

        with open(join(self.output_folder, "spec.txt"), 'w') as f:
            f.write(self.comment)


        # 2. Graph Pre-processing
        self.ground_motion_number = ground_motion_number
        self.yield_factor = None
        self.graph = None
        self.element_node_dict = None
        self.node_element_dict = None
        self.element_category_list = None
        self.each_element_category_list = None
        self.story_element_index_list = None
        self.element_length_list = None
        self.My_dict = None
        self.area_dict = None
        self.min_max_usage = None
        self.total_design_elements = None
        self.total_elements = None

        # 3. Simulator Model
        self.model_constructor_args = None
        self.model = None

        # 4. Initial Design
        self.current_design = None
        self.current_index = 0


        self._initialize_simulator_envrimoment()
        self._graph_preprocessing()
        self._initialize_simulator_model()
        self._initialize_design()

    
    def __str__(self):
        return "Structure Element Section Designer"


    def _initialize_simulator_envrimoment(self):
        print("Initializing AI-Based structural analysis simulator......")
        self.args_path = join(self.simulator_path, "training_args.json")
        self.norm_dict_path = join(self.simulator_path, "norm_dict.json")
        self.args = json.load(open(self.args_path, 'r'))
        self.norm_dict = json.load(open(self.norm_dict_path, 'r'))


    def _graph_preprocessing(self):
        print("Pre-processing structural graph......")
        self.yield_factor = self.args['yield_factor']
        self.graph, geo_info = geometry.get_graph_and_index_from_ipt(self.ipt_path, self.gm_folder, self.ground_motion_number, self.mode)
        self.element_node_dict, self.node_element_dict, self.element_category_list, self.each_element_category_list, self.element_length_list, self.story_element_index_list = geo_info
        self.graph = normalization.normalize(self.graph, self.norm_dict)
        self.My_dict = geometry.get_norm_My_dict(self.norm_dict)
        self.area_dict = geometry.get_section_area_dict(self.norm_dict)
        self.min_max_usage = geometry.get_min_max_usage(self.element_category_list, self.element_length_list, self.area_dict)
        self.total_design_elements = len(self.element_category_list)
        self.total_elements = len(self.each_element_category_list)


    def _initialize_simulator_model(self):
        self.model_constructor_args = {
            'input_dim': self.graph.x.shape[1], 'hidden_dim': self.args["hidden_dim"], 'output_dim': 15,
            'num_layers': self.args["num_layers"]}
        self.model = LSTM.LSTM(**self.model_constructor_args).to(self.device)


    def _initialize_design(self):
        self.current_design = []
        self.current_index = 0
        for i in range(self.total_design_elements):
            if self.element_category_list[i] == 1:   # this elem is beam
                self.current_design.append("W21x44")
            else:   # elem is column
                self.current_design.append("16x16x0.375")


    def _story_to_element(self, design):
        # convert from story section to element_section
        element_design = [None for _ in range(self.total_elements)]
        for i, section in enumerate(design):
            for index in range(self.story_element_index_list[i][0], self.story_element_index_list[i][1]):
                element_design[index] = section
        return element_design


    def initialize_state(self):
        self._initialize_design()


    def get_initital_state(self):
        self.initialize_design()
        return self.current_design


    def steps(self):
        return self.total_design_elements


    def get_state(self):
        return self.current_design


    def get_state_index(self):
        return self.current_index

    
    def is_final_state(self):
        return self.current_index == self.total_design_elements

    
    def available_actions(self, elem_index=None, is_beam=None):
        if elem_index != None:
            return deepcopy(available_beam_sections) if self.element_category_list[elem_index] == 1 else deepcopy(available_column_sections)
        elif is_beam != None:
            return deepcopy(available_beam_sections) if is_beam else deepcopy(available_column_sections)
        else:
            raise ValueError("Please select either elem_idnex or is_beam to get available actions.")


    def take_action(self, action):
        # check if final state already
        if self.is_final_state():
            raise ValueError("Already in final state, cannot take action anymore.")
        # action should be a beam/column section
        assert type(action) == str
        self.current_design[self.current_index] = action
        self.current_index += 1


    def design_score(self, candidate_design=None):
        if candidate_design == None:
            candidate_design = self.current_design
        if self.mode == 'story':
            candidate_design = self._story_to_element(candidate_design)
        modal_result = analysis.run_modal_analysis(self, candidate_design)
        candidate_graph = analysis.make_section_graph(self, self.graph, candidate_design, modal_result)
        response = analysis.predict(self, candidate_graph)
        score = evaluation.designScore(self, candidate_graph, response, candidate_design)
        return score


    def visualize_response(self, final_design=None):
        if final_design == None:
            final_design = self.current_design
        if self.mode == 'story':
            final_design = self._story_to_element(final_design)
        response_folder = join(self.output_folder, "Response")
        os.mkdir(response_folder)

        modal_result = analysis.run_modal_analysis(self, final_design)
        final_graph = analysis.make_section_graph(self, self.graph, final_design, modal_result)
        response = analysis.predict(self, final_graph)  # [node_num * gm_num, 2000, 15]

        node_num = self.graph.x.shape[0]
        for gm_index in range(self.ground_motion_number):
            target_level = self.graph.target_objectives[gm_index]
            print(f"Visualizing response corresponding with ground motion {gm_index+1} ({target_level})...")
            individual_folder = join(response_folder, f"GroundMotion_{gm_index+1}_{target_level}")
            os.mkdir(individual_folder)
            ground_motion = self.graph.ground_motions[:, gm_index*10 : gm_index*10 + 10]     # [2000, 10]
            output = response[node_num*gm_index:node_num*(gm_index+1), :, :]
            visualization.visualize_ground_motion(individual_folder, ground_motion, self.norm_dict, self.graph.gm_names[gm_index])
            for response_item in ["Acceleration", "Velocity", "Displacement", "Moment_Z_Column", "Moment_Z_Xbeam", "Shear_Y"]:
                visualization.visualize_response(individual_folder, self.graph.x, output, self.norm_dict, response_item)
            visualization.visualize_plasticHinge(individual_folder, self.graph.x, output, self.norm_dict)


    def output_design(self, final_design=None):
        if final_design == None:
            final_design == self.current_design
        if self.mode == 'story':
            final_design = self._story_to_element(final_design)
        geometry.reconstruct_ipt_file(self.ipt_path, self.output_path, final_design, self.node_element_dict)



