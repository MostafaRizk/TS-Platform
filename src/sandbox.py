import argparse

from agents.hardcoded.collector import HardcodedCollectorAgent
from agents.hardcoded.dropper import HardcodedDropperAgent
from agents.hardcoded.generalist import HardcodedGeneralistAgent
from fitness import FitnessCalculator
from learning.rwg import RWGLearner
import numpy as np
from agents.nn_agent_lean import NNAgent

model_name = "rwg_homogeneous_team_nn_1_1_3_1_4_8_4_1_3_7_1_3.0_0.2_2_1000_500_5_rnn_False_0_0_linear_100000_normal_0_1_6153.960000000001.npy"
parameter_filename = "single_agent.json"
fitness_calculator = FitnessCalculator(parameter_filename)

genome = NNAgent.load_model_from_file(model_name)
agent_1 = NNAgent(fitness_calculator.get_observation_size(), fitness_calculator.get_action_size(), parameter_filename, genome)
agent_2 = NNAgent(fitness_calculator.get_observation_size(), fitness_calculator.get_action_size(), parameter_filename, genome)
fitness_calculator.calculate_fitness(agent_1, agent_2, render=True, time_delay=0.1)

