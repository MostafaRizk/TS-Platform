import argparse
import copy

from agents.hardcoded.collector import HardcodedCollectorAgent
from agents.hardcoded.dropper import HardcodedDropperAgent
from agents.hardcoded.generalist import HardcodedGeneralistAgent
from agents.hardcoded.lazy_generalist import HardcodedLazyGeneralistAgent
from fitness import FitnessCalculator
from learning.rwg import RWGLearner
import numpy as np
from agents.nn_agent_lean import NNAgent

parameter_filename = "rnn_no-bias_0HL_linear.json"
fitness_calculator = FitnessCalculator(parameter_filename)
#model_name = "rwg_homogeneous_team_nn_1_2_4_1_4_8_4_1_3_7_1_3.0_0.2_2_1000_500_5_rnn_False_0_0_linear_100000_normal_0_1_74576.11999999991.npy"
model_name = "rwg_homogeneous_team_nn_1_2_3_1_4_8_4_1_3_7_1_3.0_0.2_2_1000_500_5_rnn_False_0_0_linear_100000_normal_0_1_39080.759999999966.npy"

genome = NNAgent.load_model_from_file(model_name)
agent_1 = NNAgent(fitness_calculator.get_observation_size(), fitness_calculator.get_action_size(), parameter_filename, genome)
agent_2 = NNAgent(fitness_calculator.get_observation_size(), fitness_calculator.get_action_size(), parameter_filename, genome)
fitness_calculator.calculate_fitness(agent_1, agent_2, render=True, time_delay=0.1)

"""
parameter_filename = "lazy_generalist.json"
fitness_calculator = FitnessCalculator(parameter_filename)

total_fitness = []

for seed in range(30):
    #agent_1 = HardcodedLazyGeneralistAgent(switch_probability=0.1, seed=1)
    #agent_2 = HardcodedLazyGeneralistAgent(switch_probability=0.1, seed=1)

    agent_1 = HardcodedGeneralistAgent()
    agent_2 = HardcodedGeneralistAgent()
    #agent_1 = HardcodedDropperAgent()
    #agent_2 = HardcodedCollectorAgent(seed=1)

    #results = fitness_calculator.calculate_fitness(agent_1, agent_2, render=True, time_delay=0.3)
    results = fitness_calculator.calculate_fitness(agent_1, agent_2)
    fitness_1_list = results["fitness_1_list"]
    fitness_2_list = results["fitness_2_list"]
    specialisation_list = results["specialisation_list"]
    zipped_list = zip(fitness_1_list, fitness_2_list)
    #zipped_list_2 = copy.deepcopy(zipped_list)
    #print([fitness_1 + fitness_2 for (fitness_1, fitness_2) in zipped_list_2])
    total_fitness += [np.mean([fitness_1 + fitness_2 for (fitness_1, fitness_2) in zipped_list])]

print(np.mean(total_fitness))

# 1 agent
# Generalist = 61,229 [61229.399999999936, 61229.399999999936, 61229.399999999936, 61229.399999999936, 61229.399999999936]
# Lazy Generalist (p=0.1) = 77,808 [77407.99999999997, 79407.99999999997, 77409.99999999997, 77407.99999999997, 77409.99999999997]
# p=0.05 - 71,436
# p=0.10 - 77,808
# p=0.15 - 57,413
# p=0.20 - 37,381
# p=0.50 - 19,381

# 2 agents
# Generalist = 122,458 [122458.79999999976, 122458.79999999976, 122458.79999999976, 122458.79999999976, 122458.79999999976]
# Lazy Generalist (p=0.1) = 128,025 [126828.0, 128822.4, 128823.4, 128823.4, 126828.0]

#128,025
#126,826
#128,024
#127,225
#127,624

#Long
#Generalist = 50,395
#LazyGeneralist = 60,336

#3 resources & long
#Generalist = 50,389
#LazyGeneralist = 33,477
#Specialist = 151,287

#3 resources
#Generalist = 118,472
#LazyGeneralist = 76,882
#Specialist = 156,500

#----------
# 4 & short
# Generalist = 122,458
# Lazy Generalist = 127558
# Specialist = 115,561

"""
