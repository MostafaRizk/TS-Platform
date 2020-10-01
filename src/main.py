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
Homogeneous or Het-Ind
genome = NNAgent.load_model_from_file(model_name)
agent_1 = NNAgent(fitness_calculator.get_observation_size(), fitness_calculator.get_action_size(), parameter_filename, genome)
agent_2 = NNAgent(fitness_calculator.get_observation_size(), fitness_calculator.get_action_size(), parameter_filename, genome)
results = fitness_calculator.calculate_fitness([agent_1, agent_2], render=True, time_delay=0.1)

Het-Team
full_genome = NNAgent.load_model_from_file(model_name)
mid = int(len(full_genome) / 2)
genome_part_1 = full_genome[0:mid]
genome_part_2 = full_genome[mid:]
agent_1 = NNAgent(fitness_calculator.get_observation_size(), fitness_calculator.get_action_size(), parameter_filename, genome_part_1)
agent_2 = NNAgent(fitness_calculator.get_observation_size(), fitness_calculator.get_action_size(), parameter_filename, genome_part_2)
results = fitness_calculator.calculate_fitness([agent_1, agent_2], render=True, time_delay=0.1)
"""

# Printing Fitness
"""
fitness_1_list = results["fitness_1_list"]
fitness_2_list = results["fitness_2_list"]
specialisation_list = results["specialisation_list"]
zipped_list = zip(fitness_1_list, fitness_2_list)
print(f"{np.mean([fitness_1 + fitness_2 for (fitness_1, fitness_2) in zipped_list])}")
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
scores = []
for seed in range(30):
    parameter_filename = "default_parameters.json"
    parameter_dictionary = json.loads(open(parameter_filename).read())
    parameter_dictionary['general']['seed'] = seed
    f = open(parameter_filename, "w")
    dictionary_string = json.dumps(parameter_dictionary, indent=4)
    f.write(dictionary_string)
    f.close()
    fitness_calculator = FitnessCalculator(parameter_filename)

    agent_1 = HardcodedGeneralistAgent()
    agent_2 = HardcodedGeneralistAgent()
    results = fitness_calculator.calculate_fitness(agent_1, agent_2, render=False)
    fitness_1_list = results["fitness_1_list"]
    fitness_2_list = results["fitness_2_list"]
    specialisation_list = results["specialisation_list"]
    zipped_list = zip(fitness_1_list, fitness_2_list)
    scores += np.mean([fitness_1 + fitness_2 for (fitness_1, fitness_2) in zipped_list])

print(f"{np.mean(scores)}")
"""



