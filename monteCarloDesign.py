import sys
sys.path.append('Utils/')
sys.path.append('Searching/')
sys.path.append('Models/')


from Searching import MonteCarloTreeSearch
from Searching import Agent
from Searching import Environment




# Define simulator model path and the ground motion number that will be used in seismic performance.
simulator_path = "Simulator/2022_05_24__11_56_43"
ground_motion_number = 1

# Environment
env_args = {"simulator_path": simulator_path, "ground_motion_number": ground_motion_number, "method": "MCTS"}
env = Environment.StructureSimulator(**env_args)

# Agent
agent_args = {"mode": "story", "environment": env}
agent = Agent.StructureDesigner(**agent_args)



# Use your models to get the best beam_column design list.
rounds = 100
mcts = MonteCarloTreeSearch.MCTS(agent=agent, env=env)
final_design = mcts.run(rounds)
print(f"\nFinal design: \n{final_design}\n\n")

# Output the final design to .ipt file (which can be opened by PISA3D software).
agent.output_design(final_design)

# Visualize the seismic response under design-earthquakes. (This is not nececessary!)
# agent.visualize_response(final_design)



