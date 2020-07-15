import argparse
from fitness import FitnessCalculator
from learning.rwg import RWGLearner
from learning.cma import CMALearner
from agents.hardcoded.generalist import HardcodedGeneralistAgent
from agents.hardcoded.dropper import HardcodedDropperAgent
from agents.hardcoded.collector import HardcodedCollectorAgent

'''
parser = argparse.ArgumentParser(description='Run an experiment')
parser.add_argument('--parameters', action="store", dest="parameter_filename")
parameter_filename = parser.parse_args().parameter_filename

fitness_calculator = FitnessCalculator(parameter_filename)
learner = RWGLearner(fitness_calculator)
genome, fitness = learner.learn()
'''

parameter_filename = "default_parameters.json"
fitness_calculator = FitnessCalculator(parameter_filename)

# Bootstrap
"""
learner = RWGLearner(fitness_calculator)
genome, fitness = learner.learn()
"""

# Evolve
"""
learner = CMALearner(fitness_calculator)
genome, fitness = learner.learn()
"""

# Visualise
"""
genome = NNAgent.load_model_from_file(model_name)
agent_1 = NNAgent(fitness_calculator.get_observation_size(), fitness_calculator.get_action_size(), parameter_filename, genome)
agent_2 = NNAgent(fitness_calculator.get_observation_size(), fitness_calculator.get_action_size(), parameter_filename, genome)
fitness_calculator.calculate_fitness(agent_1, agent_2, render=True, time_delay=0.1)
"""

# Specialisation
"""
for model_name in model_names:
    genome = NNAgent.load_model_from_file(model_name)
    agent_1 = NNAgent(fitness_calculator.get_observation_size(), fitness_calculator.get_action_size(), parameter_filename, genome)
    agent_2 = NNAgent(fitness_calculator.get_observation_size(), fitness_calculator.get_action_size(), parameter_filename, genome)
    results = fitness_calculator.calculate_fitness(agent_1, agent_2, render=False, time_delay=0, measure_specialisation=True)
"""

# Hardcoded
"""
agent_1 = HardcodedGeneralistAgent()
agent_2 = HardcodedGeneralistAgent()
results = fitness_calculator.calculate_fitness(agent_1, agent_2)
fitness_1_list = results["fitness_1_list"]
fitness_2_list = results["fitness_2_list"]
specialisation_list = results["specialisation_list"]
zipped_list = zip(fitness_1_list, fitness_2_list)
print(f"{np.mean([fitness_1 + fitness_2 for (fitness_1, fitness_2) in zipped_list])}")
"""



