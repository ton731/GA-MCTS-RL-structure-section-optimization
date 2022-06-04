import sys
import time
sys.path.append('Utils/')
sys.path.append('Searching/')
sys.path.append('Models/')


from Searching import GeneticAlgorithm
from Searching import Agent
from Searching import Environment




# Define simulator model path and the ground motion number that will be used in seismic performance.
simulator_path = "Simulator/2022_05_27__22_01_14"
ground_motion_number = 3

# Environment
comment = "round=100, checkpoint=5, population_size=1000, bs=8, 0.95 yield factor, beam section 7 kind"
env_args = {"simulator_path": simulator_path, "ground_motion_number": ground_motion_number, "method": "GA", "comment": comment}
env = Environment.StructureSimulator(**env_args)

# Agent
agent_args = {"mode": "story", "environment": env, "method": "GA"}
agent = Agent.StructureDesigner(**agent_args)





# Use your models to get the best beam_column design list.
rounds = 100
checkpoint = 5
start_time = time.time()
ga = GeneticAlgorithm.GA(agent=agent, env=env, rounds=rounds, checkpoint=checkpoint)
final_design = ga.run()
print(f"\nFinal design: \n{final_design}\n\n")
finish_time = time.time()
print("\n\ntotal_time: ", (finish_time - start_time)/60, "min\n\n")

# Output the final design to .ipt file (which can be opened by PISA3D software).
agent.output_design(final_design)

# Visualize the seismic response under design-earthquakes. (This is not nececessary!)
# agent.visualize_response(final_design)



