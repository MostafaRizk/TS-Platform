import argparse

from agents.hardcoded.collector import HardcodedCollectorAgent
from agents.hardcoded.dropper import HardcodedDropperAgent
from agents.hardcoded.generalist import HardcodedGeneralistAgent
from agents.hardcoded.lazy_generalist import HardcodedLazyGeneralistAgent
from fitness import FitnessCalculator
from learning.cma import CMALearner
from learning.rwg import RWGLearner
import numpy as np
from agents.nn_agent_lean import NNAgent
import pandas as pd

parameter_filename = "rnn_no-bias_0HL.json"
fitness_calculator = FitnessCalculator(parameter_filename)

model_name = "rwg_heterogeneous_team_nn_slope_1_2_4_1_4_8_4_1_3_7_1_3.0_0.2_2_1000_500_5_rnn_False_0_0_linear_100_normal_0_1_5198.959999999997.npy"
full_genome = NNAgent.load_model_from_file(model_name)

mid = int(len(full_genome) / 2)
genome_part_1 = full_genome[0:mid]
genome_part_2 = full_genome[mid:]
agent_1 = NNAgent(fitness_calculator.get_observation_size(), fitness_calculator.get_action_size(), parameter_filename, genome_part_1)
agent_2 = NNAgent(fitness_calculator.get_observation_size(), fitness_calculator.get_action_size(), parameter_filename, genome_part_2)
results = fitness_calculator.calculate_fitness(agent_1, agent_2, render=True, time_delay=0.1)
fitness_1_list = results["fitness_1_list"]
fitness_2_list = results["fitness_2_list"]
specialisation_list = results["specialisation_list"]
zipped_list = zip(fitness_1_list, fitness_2_list)
print(f"{np.mean([fitness_1 + fitness_2 for (fitness_1, fitness_2) in zipped_list])}")