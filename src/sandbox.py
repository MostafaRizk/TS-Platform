import argparse

from agents.hardcoded.collector import HardcodedCollectorAgent
from agents.hardcoded.dropper import HardcodedDropperAgent
from agents.hardcoded.generalist import HardcodedGeneralistAgent
from fitness import FitnessCalculator
from learning.rwg import RWGLearner
import numpy as np
from agents.nn_agent_lean import NNAgent

parameter_filename = "sliding_1.json"
#model_name = "rwg_homogeneous_team_nn_1_2_3_1_0_8_4_1_3_7_1_3.0_0.2_2_1000_500_5_rnn_False_0_0_linear_100000_normal_0_1_39082.27999999997.npy"
model_name = "rwg_homogeneous_team_nn_1_2_3_1_1_8_4_1_3_7_1_3.0_0.2_2_1000_500_5_rnn_False_0_0_linear_100000_normal_0_1_39879.19999999997.npy"
fitness_calculator = FitnessCalculator(parameter_filename)

genome = NNAgent.load_model_from_file(model_name)
agent_1 = NNAgent(fitness_calculator.get_observation_size(), fitness_calculator.get_action_size(), parameter_filename, genome)
agent_2 = NNAgent(fitness_calculator.get_observation_size(), fitness_calculator.get_action_size(), parameter_filename, genome)
results = fitness_calculator.calculate_fitness(agent_1, agent_2, render=True, time_delay=10) # , render=True, time_delay=0.1
fitness_1_list = results["fitness_1_list"]
fitness_2_list = results["fitness_2_list"]
specialisation_list = results["specialisation_list"]
zipped_list = zip(fitness_1_list, fitness_2_list)
print(f"{np.mean([fitness_1 + fitness_2 for (fitness_1, fitness_2) in zipped_list])}")