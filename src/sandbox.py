import argparse
import json

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

parameter_filename = "/Users/mostafa/Documents/Code/PhD/TS-Platform/results/2020_11_08_2_agents_reduced_episodes_evolution/experiments/cma_heterogeneous_team_nn_slope_4005303369_2_4_1_4_8_4_1_3_7_1_3.0_0.2_2_1000_500_5_rnn_False_1_4_tanh_100_0.2_1000_0.001_200.0.json"
fitness_calculator = FitnessCalculator(parameter_filename)
model_name = "/Users/mostafa/Documents/Code/PhD/TS-Platform/results/2020_11_08_2_agents_reduced_episodes_evolution/results/cma_heterogeneous_team_nn_slope_4005303369_2_4_1_4_8_4_1_3_7_1_3.0_0.2_2_1000_500_5_rnn_False_1_4_tanh_100_0.2_1000_0.001_200.0_95763.55999999987_final.npy"
full_genome = NNAgent.load_model_from_file(model_name)
mid = int(len(full_genome) / 2)
genome_part_1 = full_genome[0:mid]
genome_part_2 = full_genome[mid:]
agent_1 = NNAgent(fitness_calculator.get_observation_size(), fitness_calculator.get_action_size(), parameter_filename, genome_part_1)
agent_2 = NNAgent(fitness_calculator.get_observation_size(), fitness_calculator.get_action_size(), parameter_filename, genome_part_2)
results = fitness_calculator.calculate_fitness([agent_1, agent_2], render=True, time_delay=0.1)


#Genome first half(collector)
# Dropper- -974
# Collector- -1000
# Generalist- 14,968

#Genome second half (dropper)
# Dropper- -1983
# Collector- 119,430
# Generalist- 27,183