import argparse
from fitness import FitnessCalculator
from learning.rwg import RWGLearner
from agents.nn_agent import NNAgent

import sys

filenames = ["old_fitness_old_activation.json",
             "old_fitness_new_activation.json",
             "new_fitness_old_activation.json",
             "new_fitness_new_activation.json"]

model_names = ["rwg_homogeneous_team_nn_1_2_3_1_4_8_4_1_3_7_0_1.0_1.0_1_1_500_5_0_linear_5000_normal_0_1_7.6.npy",
               "rwg_homogeneous_team_nn_1_2_3_1_4_8_4_1_3_7_0_1.0_1.0_1_1_500_5_0_sigmoid_5000_normal_0_1_0.0.npy",
               "rwg_homogeneous_team_nn_1_2_3_1_4_8_4_1_3_7_1_3.0_0.2_2_1000_500_5_0_linear_5000_normal_0_1_6356.600000000001.npy",
               "rwg_homogeneous_team_nn_1_2_3_1_4_8_4_1_3_7_1_3.0_0.2_2_1000_500_5_0_sigmoid_5000_normal_0_1_-698.960000000001.npy"]

'''
for parameter_filename in filenames:
    fitness_calculator = FitnessCalculator(parameter_filename)
    learner = RWGLearner(fitness_calculator)
    genome, fitness = learner.learn()
'''

"""
0- Old fitness + old activation = 7.6
1- Old fitness + new activation = 0.0
2- New fitness + old activation = 6356.6
3- New fitness + new activation = -698.96
"""

model_id = 0
parameter_filename = filenames[model_id]
model_name = model_names[model_id]
fitness_calculator = FitnessCalculator(parameter_filename)
genome = NNAgent.load_model_from_file(model_name)
agent_1 = NNAgent(fitness_calculator.get_observation_size(), fitness_calculator.get_action_size(), parameter_filename, genome)
agent_2 = NNAgent(fitness_calculator.get_observation_size(), fitness_calculator.get_action_size(), parameter_filename, genome)
fitness_calculator.calculate_fitness(agent_1, agent_2, render=True, time_delay=0.1, )

