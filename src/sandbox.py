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

#Het-Ind
''''''
#parameter_filename = "cma_heterogeneous_individual_nn_slope_4018109722_4_8_1_4_8_8_1_3_7_1_3.0_0.2_2_1000_500_20_rnn_False_1_4_tanh_100_0.2_1000_0.001_200.0.json"
parameter_filename = "cma_heterogeneous_individual_nn_slope_550290314_4_8_1_4_8_8_1_3_7_1_3.0_0.2_2_1000_500_20_rnn_False_1_4_tanh_100_0.2_1000_0.001_200.0.json"

#model_name_1 = "cma_heterogeneous_individual_nn_slope_4018109722_4_8_1_4_8_8_1_3_7_1_3.0_0.2_2_1000_500_20_rnn_False_1_4_tanh_100_0.2_1000_0.001_200.0_3_-350.399999999998_final.npy"
model_name_1 = "cma_heterogeneous_individual_nn_slope_550290314_4_8_1_4_8_8_1_3_7_1_3.0_0.2_2_1000_500_20_rnn_False_1_4_tanh_100_0.2_1000_0.001_200.0_0_-500.0_final.npy"
model_name_2 = model_name_3 = model_name_4 = model_name_1

fitness_calculator = FitnessCalculator(parameter_filename)

genome_1 = NNAgent.load_model_from_file(model_name_1)
genome_2 = NNAgent.load_model_from_file(model_name_2)
genome_3 = NNAgent.load_model_from_file(model_name_3)
genome_4 = NNAgent.load_model_from_file(model_name_4)
agent_1 = NNAgent(fitness_calculator.get_observation_size(), fitness_calculator.get_action_size(), parameter_filename, genome_1)
agent_2 = NNAgent(fitness_calculator.get_observation_size(), fitness_calculator.get_action_size(), parameter_filename, genome_2)
agent_3 = NNAgent(fitness_calculator.get_observation_size(), fitness_calculator.get_action_size(), parameter_filename, genome_3)
agent_4 = NNAgent(fitness_calculator.get_observation_size(), fitness_calculator.get_action_size(), parameter_filename, genome_4)
#agent_0 = HardcodedDropperAgent()
#agent_1 = HardcodedCollectorAgent()
results = fitness_calculator.calculate_fitness([agent_1, agent_2, agent_3, agent_4], render=True, time_delay=0.1)#, logging=True, logfilename="rewards.csv")
mean = np.mean([results["fitness_matrix"][0][i] + results["fitness_matrix"][1][i] for i in range(len(results["fitness_matrix"][0]))])
print(mean)
print(results)


# Single agent
'''
parameter_filename = "rwg_ind_4_agents.json"
model_name = "rwg_heterogeneous_individual_nn_slope_1_1_8_1_4_8_8_1_3_7_1_3.0_0.2_2_1000_500_20_rnn_False_1_4_tanh_1000_normal_0_1_-27.890000000000136.npy"

fitness_calculator = FitnessCalculator(parameter_filename)

genome = NNAgent.load_model_from_file(model_name)
agent = NNAgent(fitness_calculator.get_observation_size(), fitness_calculator.get_action_size(), parameter_filename, genome)
results = fitness_calculator.calculate_fitness([agent], render=True, time_delay=0.1)
'''
#Het-Team
'''
parameter_filename = "cma_heterogeneous_team_nn_slope_4018109722_2_4_1_4_8_4_1_3_7_1_3.0_0.2_2_1000_500_20_rnn_False_1_4_tanh_100_0.2_1000_0.001_200.0.json"
model_name = "cma_heterogeneous_team_nn_slope_4018109722_2_4_1_4_8_4_1_3_7_1_3.0_0.2_2_1000_500_20_rnn_False_1_4_tanh_100_0.2_1000_0.001_200.0_120500.09999999986_final.npy"

#parameter_filename = "cma_heterogeneous_team_nn_slope_799981517_2_4_1_4_8_4_1_3_7_1_3.0_0.2_2_1000_500_20_rnn_False_1_4_tanh_100_0.2_1000_0.001_200.0.json"
#model_name = "cma_heterogeneous_team_nn_slope_799981517_2_4_1_4_8_4_1_3_7_1_3.0_0.2_2_1000_500_20_rnn_False_1_4_tanh_100_0.2_1000_0.001_200.0_96852.2_final.npy"

fitness_calculator = FitnessCalculator(parameter_filename)

full_genome = NNAgent.load_model_from_file(model_name)
mid = int(len(full_genome) / 2)
genome_part_1 = full_genome[0:mid]
genome_part_2 = full_genome[mid:]
#agent_1 = NNAgent(fitness_calculator.get_observation_size(), fitness_calculator.get_action_size(), parameter_filename, genome_part_1)
#agent_2 = NNAgent(fitness_calculator.get_observation_size(), fitness_calculator.get_action_size(), parameter_filename, genome_part_2)
agent_1 = HardcodedDropperAgent()
agent_2 = HardcodedCollectorAgent()
results = fitness_calculator.calculate_fitness([agent_1, agent_2], render=False, time_delay=0, logging=True, logfilename="rewards.csv")
mean = np.mean([results["fitness_matrix"][0][i] + results["fitness_matrix"][1][i] for i in range(len(results["fitness_matrix"][0]))])
print(mean)
print(results)
'''

#Genome first half(collector)
# Dropper- -974
# Collector- -1000
# Generalist- 14,968

#Genome second half (dropper)
# Dropper- -1983
# Collector- 119,430
# Generalist- 27,183