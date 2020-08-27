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

parameter_filename = "default_parameters.json"
fitness_calculator = FitnessCalculator(parameter_filename)

'''
learner = RWGLearner(fitness_calculator) 
genome, fitness = learner.learn()
'''

'''
learner = CMALearner(fitness_calculator)
genome, fitness = learner.learn()
'''

''''''
model_name = "rwg_heterogeneous_team_nn_1_2_3_1_4_8_4_1_3_7_1_3.0_0.2_2_1000_500_5_simple_rnn_False_0_0_linear_100000_normal_0_1_20812.72000000001.npy"
#model_name = "rwg_heterogeneous_team_nn_1_2_3_1_4_8_4_1_3_7_1_3.0_0.2_2_1000_500_5_simple_rnn_True_0_0_linear_100000_normal_0_1_20812.72000000001.npy"
#model_name = "rwg_heterogeneous_team_nn_1_2_3_1_4_8_4_1_3_7_1_3.0_0.2_2_1000_500_5_simple_rnn_False_1_4_tanh_100000_normal_0_1_10080.160000000003.npy"

genome = NNAgent.load_model_from_file(model_name)
agent_1 = NNAgent(fitness_calculator.get_observation_size(), fitness_calculator.get_action_size(), parameter_filename, genome)
agent_2 = NNAgent(fitness_calculator.get_observation_size(), fitness_calculator.get_action_size(), parameter_filename, genome)
fitness_calculator.calculate_fitness(agent_1, agent_2, render=True, time_delay=0.1)
