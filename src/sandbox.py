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

# Team-2. Slope. Best scoring team (specialist)
#parameter_filename = "/Users/mostafa/Documents/Code/PhD/TS-Platform/results/2020_09_25_CMA_with_shortened_seeding/experiments/cma_heterogeneous_team_nn_slope_4018109722_2_4_1_4_8_4_1_3_7_1_3.0_0.2_2_1000_500_20_rnn_False_1_4_tanh_100_0.2_1000_0.001_200.0.json"
#model_name = "/Users/mostafa/Documents/Code/PhD/TS-Platform/results/2020_09_25_CMA_with_shortened_seeding/results/cma_heterogeneous_team_nn_slope_4018109722_2_4_1_4_8_4_1_3_7_1_3.0_0.2_2_1000_500_20_rnn_False_1_4_tanh_100_0.2_1000_0.001_200.0_120500.09999999986_final.npy"

# Team-2. No slope. Best scoring team (generalist but weird)
parameter_filename = "/Users/mostafa/Documents/Code/PhD/TS-Platform/results/2020_10_07_CMA many agents and big populations/experiments/cma_heterogeneous_team_nn_slope_2252917205_2_4_1_0_8_4_1_3_7_1_1.0_1.0_2_1000_500_20_rnn_False_1_4_tanh_100_0.2_1000_0.001_200.0.json"
model_name = "/Users/mostafa/Documents/Code/PhD/TS-Platform/results/2020_10_07_CMA many agents and big populations/results/cma_heterogeneous_team_nn_slope_2252917205_2_4_1_0_8_4_1_3_7_1_1.0_1.0_2_1000_500_20_rnn_False_1_4_tanh_100_0.2_1000_0.001_200.0_39794.55_final.npy"

fitness_calculator = FitnessCalculator(parameter_filename)

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