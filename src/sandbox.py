import numpy as np
import os

from fitness import FitnessCalculator
from agents.nn_agent_lean import NNAgent

parameter_path = "/Users/mostafa/Documents/Code/PhD/TS-Platform/src/default_box_pushing_parameters.json"

fitness_calculator = FitnessCalculator(parameter_path)

num_agents = 2
agent_list = []
for i in range(num_agents):
    agent = NNAgent(fitness_calculator.get_observation_size(), fitness_calculator.get_action_size(), parameter_path)
    agent_list += [agent]

results = fitness_calculator.calculate_fitness(controller_list=agent_list, render=True, time_delay=0.1,
                                                       measure_specialisation=False, logging=False, logfilename=None,
                                                       render_mode="human")



