import numpy as np
from copy import deepcopy
from .evaluation import *
from .analysis import *



available_beam_sections = ['W21x93', 'W21x83', 'W21x73', 'W21x68', 'W21x62', 'W21x57', 'W21x50', 'W21x48', 'W21x44']
available_column_sections = ['16x16x0.875', '16x16x0.75', '16x16x0.625', '16x16x0.5', '16x16x0.375']


# https://bitcrush.medium.com/%E8%92%99%E5%9C%B0%E5%8D%A1%E7%BE%85%E6%90%9C%E7%B4%A2%E6%B3%95-ee7de77940ef


class Node:
    # Static variable
    total_elements = 0
    element_category_list, element_length_list = [], []
    element_node_dict, node_element_dict = {}, {}
    total_N = 0
    graph = None
    ipt_path, candidate_path, analysis_path = None, None, None
    My_dict, area_dict = None, None
    norm_dict = None
    min_max_usage = None
    model = None
    device = "cuda"

    def __init__(self, parent, action_section, depth, is_beam, previous_sections):
        self.parent = parent                # Last element
        self.childrens = []                 # All possible section node for next element
        self.Q = 0                          # Node total score
        self.N = 0                          # Node visited time
        self.is_beam = is_beam              # Whether it's a beam
        self.element_index = depth          # Index of all element (will be the depth of the MC Tree)
        self.next_is_beam = True if Node.element_category_list[depth+1] == 1 else False      
                                            # Check the table whether next index is beam/column
        self.untried_actions = deepcopy(available_beam_sections) if self.next_is_beam else deepcopy(available_column_sections)
                                            # Record untried section for expanding node
        self.section = action_section       # Current element's section
        self.previous_sections = previous_sections + ([self.section] if action_section else [])     # Record the section from root to this node


    @staticmethod
    def set_static(mcts_args):
        Node.total_elements = mcts_args["total_elements"]
        Node.element_category_list = [None] + mcts_args["element_category_list"] + [None]   # To make element index correspond to category, add None in the front and end
        Node.element_length_list = mcts_args["element_length_list"]
        Node.element_node_dict = mcts_args["element_node_dict"]
        Node.node_element_dict = mcts_args["node_element_dict"]
        Node.graph = mcts_args["graph"]
        Node.ipt_path = mcts_args["ipt_path"]
        Node.candidate_path = mcts_args["candidate_path"]
        Node.analysis_path = mcts_args["analysis_path"]
        Node.My_dict = mcts_args["My_dict"]
        Node.area_dict = mcts_args["area_dict"]
        Node.norm_dict = mcts_args["norm_dict"]
        Node.min_max_usage = mcts_args["min_max_usage"]
        Node.model = mcts_args["model"]

        print(f"There are total {Node.total_elements} elements.")
        print(f"Element Category List: {Node.element_category_list[:10]}...")
        print(f"Graph: {Node.graph}")
        print(f"Input path: {Node.ipt_path}")
        print(f"Candidate path: {Node.candidate_path}")
        print(f"Analysis path: {Node.analysis_path}")
        print('\n'*3)

    @staticmethod
    def get_random_action(is_beam):
        available_actions = available_beam_sections if is_beam else available_column_sections
        return np.random.choice(available_actions)

    def weight_func(self, c_param=1.4):
        if self.N != 0:
            w = self.Q / self.N + c_param * np.sqrt(np.log(Node.total_N) / self.N)
        else:
            w = 0.0
        return w

    def select(self, c_param=1.4):
        # print(f"Select, depth:{self.element_index}, {self.previous_sections}")
        weights = [child_node.weight_func(c_param) for child_node in self.childrens]
        action = np.argmax(weights)
        next_node = self.childrens[action]
        return next_node

    def expand(self):
        # print(f"Expand, depth:{self.element_index}, {self.previous_sections}")
        if len(self.untried_actions) == 0: return
        action = self.untried_actions.pop()
        child_node = Node(self, action, self.element_index+1, self.next_is_beam, self.previous_sections)
        self.childrens.append(child_node)
        return child_node

    def update(self, score):
        self.N += 1
        self.Q += score
        if self.is_root_node() == False:
            self.parent.update(score)

    def rollout(self):
        # print(f"Rollout, depth:{self.element_index}, {self.previous_sections}")
        Node.total_N += 1
        current_node = deepcopy(self)
        while current_node.element_index < Node.total_elements:
            # random_rollout_action = Node.get_random_action(current_node.next_is_beam)
            # next_node = Node(None, random_rollout_action, current_node.element_index+1, current_node.next_is_beam, current_node.previous_sections)
            rollout_action = "W21x44" if current_node.next_is_beam else "16x16x0.375"
            next_node = Node(None, rollout_action, current_node.element_index+1, current_node.next_is_beam, current_node.previous_sections)
            current_node = next_node

        assert current_node.element_index == Node.total_elements

        # Now we have all element's section, make it as a graph and feed into LSTM to get the design score
        candidate_design = current_node.previous_sections
        modal_result = run_modal_analysis(self, candidate_design)
        candidate_graph = make_section_graph(self, Node.graph, candidate_design, modal_result)
        response = predict(self, candidate_graph)
        score = designScore(self, candidate_graph, response, candidate_design)
        return score

    def is_full_expand(self):
        return len(self.untried_actions) == 0

    def has_children(self):
        return len(self.childrens) != 0

    def is_root_node(self):
        return self.element_index == 0


    


class MCTS:
    # def __init__(self, total_elements, element_category_list, graph, candidate_path, My_dict):
    def __init__(self, mcts_args):
        self.root = None
        self.current_node = None
        Node.set_static(mcts_args)

    def __str__(self):
        return "Monte Carlo Tree Search AI"

    def simulation(self, times=1000):
        for i in range(times):
            leaf_node = self.simulation_policy()
            score = leaf_node.rollout()
            leaf_node.update(score)
            print(f"Simulation: {i+1}, depth: {leaf_node.element_index}, score: {score:.4f}, sections: {leaf_node.previous_sections}")

    def simulation_policy(self):
        current_node = self.current_node
        while current_node.element_index < Node.total_elements:
            if current_node.is_full_expand():
                current_node = current_node.select()
            else:
                return current_node.expand()
        leaf_node = current_node
        return leaf_node

    def inference(self):
        current_node = self.root
        # Expand to the current best leaf node
        while(current_node.has_children()):
            current_node = current_node.select(c_param=0)
        # If leaf node doesn't finish (not all element are assigned section), then rollout through the end
        while current_node.element_index < Node.total_elements:
            # random_rollout_action = Node.get_random_action(current_node.next_is_beam)
            # next_node = Node(None, random_rollout_action, current_node.element_index+1, current_node.next_is_beam, current_node.previous_sections)
            rollout_action = "W21x44" if current_node.next_is_beam else "16x16x0.375"
            next_node = Node(None, rollout_action, current_node.element_index+1, current_node.next_is_beam, current_node.previous_sections)
            current_node = next_node
        return current_node.previous_sections


    def take_action(self, times=1000):
        # Initialize root node
        self.root = Node(None, None, 0, 0, [])
        self.current_node = self.root

        # Simulation
        self.simulation(times)

        # Inference
        result = self.inference()

        return result













