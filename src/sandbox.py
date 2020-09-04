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

model_name = "rwg_heterogeneous_team_nn_slope_1_2_4_1_4_8_4_1_3_7_1_3.0_0.2_2_1000_500_5_rnn_False_0_0_tanh_20000_normal_0_1_16268.08000000001.npy"
parameter_filename = "rnn_no-bias_0HL_tanh.json"
fitness_calculator = FitnessCalculator(parameter_filename)

"""
csv_file = "sorted_rwg.csv"
f = open(csv_file, "r")
genomes = f.read().strip().split("\n")
f.close()

csv_genome = np.array([float(weight) for weight in genomes[0].split(",")[0:-6]])
csv_mid = int(len(csv_genome) / 2)
genome_part_1_csv = csv_genome[0:csv_mid]
genome_part_2_csv = csv_genome[csv_mid:]
"""

npy_genome = NNAgent.load_model_from_file(model_name)
npy_mid = int(len(npy_genome) / 2)
genome_part_1_npy = npy_genome[0:npy_mid]
genome_part_2_npy = npy_genome[npy_mid:]
agent_1 = NNAgent(fitness_calculator.get_observation_size(), fitness_calculator.get_action_size(), parameter_filename, genome_part_1_npy)
agent_2 = NNAgent(fitness_calculator.get_observation_size(), fitness_calculator.get_action_size(), parameter_filename, genome_part_2_npy)
results = fitness_calculator.calculate_fitness(agent_1, agent_2, render=True, time_delay=0.1)