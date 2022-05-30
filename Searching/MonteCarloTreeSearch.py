import numpy as np
from copy import deepcopy




# Source: https://bitcrush.medium.com/%E8%92%99%E5%9C%B0%E5%8D%A1%E7%BE%85%E6%90%9C%E7%B4%A2%E6%B3%95-ee7de77940ef


class Node:
    # Static Variables
    total_N = 0
    agent = None
    env = None

    def __init__(self, parent, action_section, depth, is_beam, previous_sections):
        self.parent = parent                # Last element
        self.childrens = []                 # All possible section node for next element
        self.Q = 0                          # Node total score
        self.N = 0                          # Node visited time
        self.is_beam = is_beam              # Whether it's a beam
        self.element_index = depth          # Index of all element (will be the depth of the MC Tree)
        self.next_is_beam = True if Node.agent.element_category_list[depth] == 1 else False      
                                            # Check the table whether next index is beam/column, if current depth=2, then next depth=3, the third elem, where index is 2 in list
        self.untried_actions = Node.agent.available_actions(is_beam=self.next_is_beam)
                                            # Record untried section for expanding node
        self.section = action_section       # Current element's section
        self.current_design = deepcopy(previous_sections)
        if depth > 0:   # update section, for depth = 2, it's the second element which index in the design is 1.
            self.current_design[depth-1] = self.section


    @staticmethod
    def set_static(agent, env):
        Node.agent = agent
        Node.env = env

        print(f"There are total {agent.total_elements} elements.")
        print(f"Element Category List: {agent.element_category_list[:10]}...")
        print(f"Graph: {agent.graph}")
        print(f"Input path: {agent.ipt_path}")
        print(f"Analysis path: {env.analysis_path}")
        print(f"Output path: {env.output_path}")
        print(f"Initial design: \n{agent.get_state()}")
        print('\n'*3)


    def weight_func(self, c_param=1.4):
        if self.N != 0:
            w = self.Q / self.N + c_param * np.sqrt(np.log(Node.total_N) / self.N)
        else:
            w = 0.0
        return w

    def select(self, c_param=1.0):
        # print(f"Select, depth:{self.element_index}, {self.previous_sections}")
        weights = [child_node.weight_func(c_param) for child_node in self.childrens]
        action = np.argmax(weights)
        next_node = self.childrens[action]
        return next_node

    def expand(self):
        # print(f"Expand, depth:{self.element_index}, {self.previous_sections}")
        if len(self.untried_actions) == 0: return
        action = self.untried_actions.pop()
        child_node = Node(self, action, self.element_index+1, self.next_is_beam, self.current_design)
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

        # Now we have all element's section, make it as a graph and feed into LSTM to get the design score
        candidate_design = self.current_design
        score = Node.env.score(Node.agent, candidate_design)
        return score

    def is_full_expand(self):
        return len(self.untried_actions) == 0

    def has_children(self):
        return len(self.childrens) != 0

    def is_root_node(self):
        return self.element_index == 0


    


class MCTS:
    def __init__(self, agent, env):
        self.root = None
        self.current_node = None
        Node.set_static(agent, env)

    def __str__(self):
        return "Monte Carlo Tree Search AI"

    def simulation(self, times=1000, checkpoint=500):
        for i in range(times):
            leaf_node = self.simulation_policy()
            depth = leaf_node.element_index
            print(f"Simulation: {i+1:5d}, depth: {depth:3d}, score: ", end="")
            score = leaf_node.rollout()
            leaf_node.update(score)
            print(f"{score:.4f}, sections: {leaf_node.current_design[:depth]}")
            if (i+1) % checkpoint == 0:
                checkpoint_result = self.inference()
                Node.agent.output_design(checkpoint_result, str(i+1)) 

    def simulation_policy(self):
        current_node = self.current_node
        while current_node.element_index < Node.agent.steps():
            if current_node.is_full_expand():
                current_node = current_node.select()
            else:
                return current_node.expand()
        leaf_node = current_node
        return leaf_node

    def inference(self):
        current_node = self.root
        # Expand to the current best leaf node
        # If leaf node doesn't finish (not all element are assigned section), then use the default thinnest design
        while(current_node.has_children()):
            current_node = current_node.select(c_param=0)
    
        return current_node.current_design


    def run(self, times=1000, checkpoint=500):
        # Initialize root node
        Node.agent.initialize_state()
        self.root = Node(None, None, 0, 0, Node.agent.get_state())
        self.current_node = self.root

        # Simulation
        self.simulation(times, checkpoint)

        # Inference
        result = self.inference()

        return result













