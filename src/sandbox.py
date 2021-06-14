import numpy as np
import os

from fitness import FitnessCalculator
from agents.nn_agent_lean import NNAgent

def visualise(model_type, directory, parameter_file, model_file):
    #parameter_path = "/Users/mostafa/Documents/Code/PhD/TS-Platform/results/2021_06_01_tmaze_rwg/data/ffnn_bias_0HL_tanh.json"
    parameter_path = os.path.join(directory, parameter_file)
    #parameter_path = "ffnn_bias_0HL_tanh.json"
    fitness_calculator = FitnessCalculator(parameter_path)
    model_path = os.path.join(directory, model_file)
    full_genome = np.load(model_path, allow_pickle=True)
    num_agents = 2

    if model_type == "centralised":
        controller_list = [None] * num_agents

        for i in range(num_agents):
            start = i * int(len(full_genome) / num_agents)
            end = (i + 1) * int(len(full_genome) / num_agents)
            sub_genome = full_genome[start:end]
            agent = NNAgent(fitness_calculator.get_observation_size(), fitness_calculator.get_action_size(),
                            parameter_path, sub_genome)

            controller_list[i] = agent

    else:
        controller_list = [NNAgent(fitness_calculator.get_observation_size()*num_agents, fitness_calculator.get_action_size()*num_agents,
                        parameter_path, full_genome)]

    rendering = True
    time_delay = 1

    results = fitness_calculator.calculate_fitness(controller_list=controller_list, render=rendering, time_delay=time_delay,
                                                       measure_specialisation=True, logging=False, logfilename=None,
                                                       render_mode="human")

    print(results)

directory = "/Users/mostafa/Documents/Code/PhD/TS-Platform/src"
parameter_file = "default_tmaze_parameters_rwg_fc.json"
model_file = "fully-centralised_rwg_heterogeneous_team_nn_tmaze_1_2_9_3_10_1_ffnn_True_0_0_tanh_20000_normal_0_1_20.0.npy"

visualise("fully-centralised", directory, parameter_file, model_file)