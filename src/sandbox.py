import argparse

from agents.hardcoded.collector import HardcodedCollectorAgent
from agents.hardcoded.dropper import HardcodedDropperAgent
from agents.hardcoded.generalist import HardcodedGeneralistAgent
from fitness import FitnessCalculator
from learning.rwg import RWGLearner
import numpy as np

parameter_filename = "default_parameters.json"
fitness_calculator = FitnessCalculator(parameter_filename)

agent_1 = HardcodedGeneralistAgent()
agent_2 = HardcodedGeneralistAgent()
results = fitness_calculator.calculate_fitness(agent_1, agent_2)
fitness_1_list = results["fitness_1_list"]
fitness_2_list = results["fitness_2_list"]
specialisation_list = results["specialisation_list"]
zipped_list = zip(fitness_1_list, fitness_2_list)
print(f"{np.mean([fitness_1 + fitness_2 for (fitness_1, fitness_2) in zipped_list])}")

