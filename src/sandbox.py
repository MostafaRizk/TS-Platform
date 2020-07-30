import argparse

from agents.hardcoded.collector import HardcodedCollectorAgent
from agents.hardcoded.dropper import HardcodedDropperAgent
from agents.hardcoded.generalist import HardcodedGeneralistAgent
from agents.hardcoded.lazy_generalist import HardcodedLazyGeneralistAgent
from fitness import FitnessCalculator
from learning.rwg import RWGLearner
import numpy as np
from agents.nn_agent_lean import NNAgent

parameter_filename = "single_agent.json"
fitness_calculator = FitnessCalculator(parameter_filename)

agent_1 = HardcodedLazyGeneralistAgent(switch_probability=0.1, seed=1)
agent_2 = HardcodedLazyGeneralistAgent(switch_probability=0.1, seed=1)
results = fitness_calculator.calculate_fitness(agent_1, agent_2, render=True, time_delay=0.1)
#results = fitness_calculator.calculate_fitness(agent_1, agent_2)
fitness_1_list = results["fitness_1_list"]
fitness_2_list = results["fitness_2_list"]
specialisation_list = results["specialisation_list"]
zipped_list = zip(fitness_1_list, fitness_2_list)
print(f"{np.mean([fitness_1 + fitness_2 for (fitness_1, fitness_2) in zipped_list])}")