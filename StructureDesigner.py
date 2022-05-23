import torch
import json
from copy import deepcopy

from Utils import geometry
from Utils import normalization
from Models import LSTM
from Utils import analysis
from Utils import evaluation



available_beam_sections = ['W21x93', 'W21x83', 'W21x73', 'W21x68', 'W21x62', 'W21x57', 'W21x50', 'W21x48', 'W21x44']
available_column_sections = ['16x16x0.875', '16x16x0.75', '16x16x0.625', '16x16x0.5', '16x16x0.375']



class StructureDesigner:
    def __init__(self, simulator_path, simulation_processing_dir="Files"):
        # 1. Simulator Enviroment
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.simulator_path = simulator_path
        self.args_path = None
        self.norm_dict_path = None
        self.args = None
        self.norm_dict = None
        self.simulation_processing_dir = simulation_processing_dir
        self.ipt_path = self.simulation_processing_dir + "/Input/structure.ipt"
        self.gm_path = self.simulation_processing_dir + "/Input/ground_motion.txt"
        self.candidate_path = self.simulation_processing_dir + "/Candidate/candidate.ipt"
        self.analysis_path = self.simulation_processing_dir + "/Analysis/modal.ipt"
        self.output_path = self.simulation_processing_dir + "/Output/design.ipt"

        # 2. Graph Pre-processing
        self.yield_factor = None
        self.graph = None
        self.element_node_dict = None
        self.node_element_dict = None
        self.element_category_list = None
        self.element_length_list = None
        self.My_dict = None
        self.area_dict = None
        self.min_max_usage = None
        self.total_elements = None

        # 3. Simulator Model
        self.model_constructor_args = None
        self.model = None

        # 4. Initial Design
        self.current_design = None


        self._initialize_simulator_envrimoment()
        self._graph_preprocessing()
        self._initialize_simulator_model()
        self._initialize_design()

    
    def __str__(self):
        return "Structure Element Section Designer"


    def _initialize_simulator_envrimoment(self):
        print("Initializing AI-Based structural analysis simulator......")
        self.args_path = self.simulator_path + "training_args.json"
        self.norm_dict_path = self.simulator_path + "norm_dict.json"
        self.args = json.load(open(self.args_path, 'r'))
        self.norm_dict = json.load(open(self.norm_dict_path, 'r'))

    def _graph_preprocessing(self):
        print("Pre-processing structural graph......")
        self.yield_factor = self.args['yield_factor']
        self.graph, geo_info = geometry.get_graph_and_index_from_ipt(self.ipt_path, self.gm_path)
        self.element_node_dict, self.node_element_dict, self.element_category_list, self.element_length_list = geo_info
        self.graph = normalization.normalize(self.graph, self.norm_dict, self.yield_factor)
        self.My_dict = geometry.get_norm_My_dict(self.norm_dict)
        self.area_dict = geometry.get_section_area_dict(self.norm_dict)
        self.min_max_usage = geometry.get_min_max_usage(self.element_category_list, self.element_length_list, self.area_dict)
        self.total_elements = len(self.element_category_list)

    def _initialize_simulator_model(self):
        self.model_constructor_args = {
            'input_dim': self.graph.x.shape[1], 'hidden_dim': self.args["hidden_dim"], 'output_dim': 15,
            'num_layers': self.args["num_layers"]}
        self.model = LSTM.LSTM(**self.model_constructor_args).to(self.device)

    def _initialize_design(self):
        self.current_design = []
        for i in range(self.total_elements):
            if self.element_category_list[i] == 1:   # this elem is beam
                self.current_design.append("W21x44")
            else:   # elem is column
                self.current_design.append("16x16x0.375")

    
    def available_actions(self, elem_index=None, is_beam=None):
        if elem_index != None:
            return deepcopy(available_beam_sections) if self.element_category_list[elem_index] == 1 else deepcopy(available_column_sections)
        elif is_beam != None:
            return deepcopy(available_beam_sections) if is_beam else deepcopy(available_column_sections)
        else:
            raise ValueError("Please select either elem_idnex or is_beam to get available actions.")

    def initialize_design(self):
        self._initialize_design()

    def get_initital_design(self):
        self.initialize_design()
        return self.current_design

    def steps(self):
        return self.total_elements

    def designScore(self, candidate_design=None):
        modal_result = analysis.run_modal_analysis(self, candidate_design)
        candidate_graph = analysis.make_section_graph(self, self.graph, candidate_design, modal_result)
        response = analysis.predict(self, candidate_graph)
        score = evaluation.designScore(self, candidate_graph, response, candidate_design)
        return score

    def output_design(self, final_design=None):
        geometry.reconstruct_ipt_file(self.ipt_path, self.output_path, final_design, self.node_element_dict)



