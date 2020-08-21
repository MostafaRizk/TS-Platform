import argparse

from agents.hardcoded.collector import HardcodedCollectorAgent
from agents.hardcoded.dropper import HardcodedDropperAgent
from agents.hardcoded.generalist import HardcodedGeneralistAgent
from agents.hardcoded.lazy_generalist import HardcodedLazyGeneralistAgent
from fitness import FitnessCalculator
from learning.rwg import RWGLearner
import numpy as np
from agents.nn_agent_lean import NNAgent

parameter_filename = "default_parameters.json"
fitness_calculator = FitnessCalculator(parameter_filename)

model_name = "rwg_heterogeneous_team_nn_1_2_3_1_6_8_4_1_3_7_1_3.0_0.2_2_1000_500_5_rnn_False_0_4_linear_1000_normal_0_1_1223.2800000000002.npy"

genome = NNAgent.load_model_from_file(model_name)
agent_1 = NNAgent(fitness_calculator.get_observation_size(), fitness_calculator.get_action_size(), parameter_filename, genome)
agent_2 = NNAgent(fitness_calculator.get_observation_size(), fitness_calculator.get_action_size(), parameter_filename, genome)
fitness_calculator.calculate_fitness(agent_1, agent_2, render=True, time_delay=0.1)