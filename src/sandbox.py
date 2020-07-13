from agents.hardcoded.collector import HardcodedCollectorAgent
from agents.hardcoded.dropper import HardcodedDropperAgent
from agents.hardcoded.generalist import HardcodedGeneralistAgent
from agents.nn_agent import NNAgent
from fitness import FitnessCalculator
from learning.rwg import RWGLearner

"""
parameter_filename = "cma_homogeneous_team_nn_117628830_2_3_1_4_8_4_1_3_7_1_3.0_0.2_2_1000_500_5_0_linear_80_0.2_5000_0.001_200.0.json"
fitness_calculator = FitnessCalculator(parameter_filename)

model_name = "cma_homogeneous_team_nn_117628830_2_3_1_4_8_4_1_3_7_1_3.0_0.2_2_1000_500_5_0_linear_80_0.2_5000_0.001_200.0_88131.83999999985_final.npy"
"""

#genome = NNAgent.load_model_from_file(model_name)
#agent_1 = NNAgent(fitness_calculator.get_observation_size(), fitness_calculator.get_action_size(), parameter_filename, genome)
#agent_2 = NNAgent(fitness_calculator.get_observation_size(), fitness_calculator.get_action_size(), parameter_filename, genome)
#fitness_calculator.calculate_fitness(agent_1, agent_2, render=True, time_delay=0.1)

"""
agent_1 = HardcodedDropperAgent()
agent_2 = HardcodedCollectorAgent()
results = fitness_calculator.calculate_fitness(agent_1, agent_2, render=True, time_delay=0.1)
fitness_1 = results["fitness_1"]
fitness_2 = results["fitness_2"]
specialisation = results["specialisation"]
print(f"Team fitness: {fitness_1+fitness_2}")
"""

import argparse
from fitness import FitnessCalculator
from learning.rwg import RWGLearner

parser = argparse.ArgumentParser(description='Run an experiment')
parser.add_argument('--parameters', action="store", dest="parameter_filename")
parameter_filename = parser.parse_args().parameter_filename

fitness_calculator = FitnessCalculator(parameter_filename)
learner = RWGLearner(fitness_calculator)
genome, fitness = learner.learn()