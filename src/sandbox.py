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
    time_delay = 0.2

    results = fitness_calculator.calculate_fitness(controller_list=controller_list, render=rendering, time_delay=time_delay,
                                                       measure_specialisation=True, logging=False, logfilename=None,
                                                       render_mode="human")

    print(results)

#directory = "/Users/mostafa/Documents/Code/PhD/TS-Platform/results/2021_07_01_a_slope_experiments/data/"
#parameter_file = "centralised_cma_heterogeneous_team_nn_slope_550290314_2_16_1_8_8_16_1_3_7_1_3.0_0.2_2_1000_100_5_False_rnn_False_1_4_tanh_100_0.2_500_0.001_0.0_1000_0.0.json"
#model_file = "centralised_cma_heterogeneous_team_nn_slope_550290314_2_16_1_8_8_16_1_3_7_1_3.0_0.2_2_1000_100_5_False_rnn_False_1_4_tanh_100_0.2_500_0.001_0.0_1000_0.0_5666.160000000001_final.npy"

#visualise("centralised", directory, parameter_file, model_file)


def visualise_agents(parameter_file, model_list):
    rendering = True
    time_delay = 0.1
    fitness_calculator = FitnessCalculator(parameter_file)
    controller_list = []

    for model in model_list:
        agent_model = np.load(model, allow_pickle=True)
        controller_list += [NNAgent(fitness_calculator.get_observation_size(), fitness_calculator.get_action_size(),
                parameter_file, agent_model)]


    results = fitness_calculator.calculate_fitness(controller_list=controller_list, render=rendering,
                                                   time_delay=time_delay,
                                                   measure_specialisation=True, logging=False, logfilename=None,
                                                   render_mode="human")
    print(results)


#parameter_file = "/Users/mostafa/Documents/Code/PhD/TS-Platform/results/2021_07_02_a_slope_more_novelty/data/decentralised_cma_heterogeneous_individual_nn_slope_1904615677_2_16_1_8_8_16_1_3_7_1_3.0_0.2_2_1000_100_5_False_rnn_False_1_4_tanh_100_0.2_500_0.001_0.0_1000_0.0.json"
#model1 = "/Users/mostafa/Documents/Code/PhD/TS-Platform/results/2021_07_02_a_slope_more_novelty/data/decentralised_cma_heterogeneous_individual_nn_slope_1904615677_2_16_1_8_8_16_1_3_7_1_3.0_0.2_2_1000_100_5_False_rnn_False_1_4_tanh_100_0.2_500_0.001_0.0_1000_0.0_0_-26.160000000000004_0.npy"
#model2 = "/Users/mostafa/Documents/Code/PhD/TS-Platform/results/2021_07_02_a_slope_more_novelty/data/decentralised_cma_heterogeneous_individual_nn_slope_1904615677_2_16_1_8_8_16_1_3_7_1_3.0_0.2_2_1000_100_5_False_rnn_False_1_4_tanh_100_0.2_500_0.001_0.0_1000_0.0_1_278.44_0.npy"
#model_list = [model1, model2]

#visualise_agents(parameter_file, model_list)

#################
from deap import base, creator, algorithms, tools

genome_length = 10
seed = 1
np_random = np.random.RandomState(seed)

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_float", np_random.random)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=genome_length)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

algorithms.eaSteadyState(toolbox, population=100, ngen=1000)