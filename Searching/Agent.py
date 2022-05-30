import os
from copy import deepcopy

from Utils import geometry
from Utils import normalization




available_beam_sections = ['W21x93', 'W21x83', 'W21x73', 'W21x68', 'W21x62', 'W21x57', 'W21x50', 'W21x48', 'W21x44']
available_column_sections = ['16x16x0.875', '16x16x0.75', '16x16x0.625', '16x16x0.5', '16x16x0.375']



class StructureDesigner:
    def __init__(self, mode='element', environment=None):

        # 1. Mode selection and environment
        self.mode = mode    # element / story
        self.env = environment
        
        
        # 2. Graph Pre-processing
        self.simulation_processing_dir = "Files"
        self.ipt_path = None
        self.graph = None
        self.element_node_dict = None
        self.node_element_dict = None
        self.element_category_list = None
        self.each_element_category_list = None
        self.story_element_index_list = None
        self.element_length_list = None
        self.node_drift_node_dict = None
        self.story_num = None
        self.min_max_usage = None
        self.total_design_elements = None
        self.total_elements = None

        
        # 3. Initial Design
        self.current_design = None
        self.current_index = 0


        # 4. Initialization
        self._graph_preprocessing()
        self._initialize_design()

    
    def __str__(self):
        return "Structure Element Section Designer"


    def _graph_preprocessing(self):
        print("Pre-processing structural graph......")
        self.ipt_path = os.path.join(self.simulation_processing_dir, "Input/structure.ipt")
        self.graph, geo_info = geometry.get_graph_and_index_from_ipt(self.ipt_path, self.mode, self.env.ground_motion_number)
        self.element_node_dict, self.node_element_dict, self.element_category_list, self.each_element_category_list, self.element_length_list, self.story_element_index_list, self.node_drift_node_dict = geo_info
        self.story_num = len(self.node_drift_node_dict.keys())
        print(f"The structure has {self.story_num} stories.")
        self.graph = normalization.normalize(self.graph, self.env.norm_dict)
        self.min_max_usage = geometry.get_min_max_usage(self.each_element_category_list, self.element_length_list, self.env.area_dict)
        self.total_design_elements = len(self.element_category_list)
        self.total_elements = len(self.each_element_category_list)


    def _initialize_design(self):
        self.current_design = []
        self.current_index = 0
        for i in range(self.total_design_elements):
            if self.element_category_list[i] == 1:   # this elem is beam
                self.current_design.append("W21x44")
            else:   # elem is column
                self.current_design.append("16x16x0.375")


    


    def initialize_state(self):
        self._initialize_design()


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


    # The decorator which convert the design to element design first.
    def _story_to_element(callback):
        def run(self, design=None, checkpoint_name=None):
            if design == None:
                design = self.current_design
            if len(design) != self.total_elements:
                element_design = [None for _ in range(self.total_elements)]
                for i, section in enumerate(design):
                    for index in range(self.story_element_index_list[i][0], self.story_element_index_list[i][1]):
                        element_design[index] = section
                design = element_design
            return callback(self, design, checkpoint_name)
        return run
            

    @_story_to_element
    def get_design(self, design, _):
        return design

    
    @_story_to_element
    def output_design(self, final_design, checkpoint_name):
        if checkpoint_name != None:
            output_path = os.path.join(self.env.output_folder, f"{checkpoint_name}.ipt")
        else:
            output_path = self.env.output_path
        geometry.reconstruct_ipt_file(self.ipt_path, output_path, final_design, self.node_element_dict)


    @_story_to_element
    def visualize_response(self, final_design, _):
        self.env.visualize_response(self, final_design)





    # Try these with decorator
    '''
    def _story_to_element(self, design):
        # convert from story section to element_section
        element_design = [None for _ in range(self.total_elements)]
        for i, section in enumerate(design):
            for index in range(self.story_element_index_list[i][0], self.story_element_index_list[i][1]):
                element_design[index] = section
        return element_design


    # Convert a 'story' mode stroy design to element design
    def get_design(self, design=None):
        if design == None:
            design = self.current_design
        return self._story_to_element(design) if self.mode == 'story' else design


    def output_design(self, final_design=None):
        if final_design == None:
            final_design == self.current_design
        if len(final_design) != self.total_elements:
            final_design = self._story_to_element(final_design)
        geometry.reconstruct_ipt_file(self.ipt_path, self.env.output_path, final_design, self.node_element_dict)

    
    def visualize_response(self, final_design=None):
        if final_design == None:
            final_design == self.current_design
        if len(final_design) != self.total_elements:
            final_design = self._story_to_element(final_design)
        self.env.visualize_response(self, final_design)
    '''



