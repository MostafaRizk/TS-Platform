import numpy as np

from fitness import FitnessCalculator
from agents.nn_agent_lean import NNAgent

#parameter_path = "/Users/mostafa/Documents/Code/PhD/TS-Platform/results/2021_06_01_tmaze_rwg/data/ffnn_bias_0HL_tanh.json"
parameter_path = "/Users/mostafa/Documents/Code/PhD/TS-Platform/results/2021_06_04_tmaze_obstacles_scalability/data/ffnn_bias_0HL_tanh.json"
fitness_calculator = FitnessCalculator(parameter_path)
model_path = "/Users/mostafa/Documents/Code/PhD/TS-Platform/src/centralised_cma_heterogeneous_team_nn_tmaze_1_2_9_3_10_5_ffnn_True_0_4_tanh_100_0.2_1000_0.001_0.0_1000_0.0_18.0_final.npy"
full_genome = np.load(model_path)
num_agents = 2
agent_list = [None] * num_agents

for i in range(num_agents):
    start = i * int(len(full_genome) / num_agents)
    end = (i + 1) * int(len(full_genome) / num_agents)
    sub_genome = full_genome[start:end]
    agent = NNAgent(fitness_calculator.get_observation_size(), fitness_calculator.get_action_size(),
                    parameter_path, sub_genome)

    agent_list[i] = agent

rendering = True
time_delay = 1

results = fitness_calculator.calculate_fitness(agent_list=agent_list, render=rendering, time_delay=time_delay,
                                                   measure_specialisation=True, logging=False, logfilename=None,
                                                   render_mode="human")

print(results)

