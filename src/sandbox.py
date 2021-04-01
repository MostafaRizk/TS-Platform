import json
import numpy as np

from qdpy.algorithms import *
from qdpy.containers import *
from qdpy.benchmarks import *
from qdpy.plots import *
from qdpy.base import *
from qdpy import tools

from fitness import FitnessCalculator
from agents.nn_agent_lean import NNAgent
from operator import add

param_file = "default_parameters.json"
parameter_dictionary = json.loads(open(param_file).read())
num_agents = 2
num_generations = 1000
pop_size = 100
fitness_calculator = FitnessCalculator(param_file)
genome_length = NNAgent(fitness_calculator.get_observation_size(), fitness_calculator.get_action_size(), param_file).get_num_weights() * num_agents


def eval_fn(full_genome):
    agent_list = []

    for i in range(num_agents):
        start = i * int(len(full_genome) / num_agents)
        end = (i + 1) * int(len(full_genome) / num_agents)
        sub_genome = full_genome[start:end]
        agent = NNAgent(fitness_calculator.get_observation_size(), fitness_calculator.get_action_size(), param_file, sub_genome)
        agent_list += [agent]

    results = fitness_calculator.calculate_fitness(agent_list)
    team_fitness_list = [0] * len(results['fitness_matrix'][0])

    for j in range(num_agents):
        team_fitness_list = list(map(add, team_fitness_list, results['fitness_matrix'][j]))

    team_score = np.mean(team_fitness_list)
    features = tuple(np.mean(results['behaviour_characterisation_list'], axis=0))
    return (-team_score,), features

# MAP-Elites with CMA
grid_shape = (8, 8) #tuple([10 for i in range(genome_length)])
weight_range = ( (0, 7), (0, 7) ) #tuple([(-6.0, 6.0) for _ in range(genome_length)])

grid = containers.Grid(shape=grid_shape, max_items_per_bin=1, fitness_domain=((-100000, 100000),),
                       features_domain=weight_range)

#algo = algorithms.RandomSearchMutPolyBounded(grid, budget=num_generations, batch_size=pop_size, dimension=genome_length, optimisation_task="maximisation")

#self.es = cma.CMAEvolutionStrategy([0.] * self.dimension, self.sigma0, self._opts)
options = {'seed': parameter_dictionary['general']['seed'],
           #'maxiter': parameter_dictionary['algorithm']['cma']['generations'],
           'maxiter': num_generations,
           #'popsize': self.get_genome_population_length(),
           'popsize': pop_size,
           'tolx': parameter_dictionary['algorithm']['cma']['tolx'],
           'tolfunhist': parameter_dictionary['algorithm']['cma']['tolfunhist'],
           'tolflatfitness': parameter_dictionary['algorithm']['cma']['tolflatfitness'],
           'tolfun': parameter_dictionary['algorithm']['cma']['tolfun']}

algo = algorithms.CMAES(grid, budget=num_generations, dimension=genome_length, sigma0=0.8)

best = algo.optimise(eval_fn, batch_mode=False)

print(eval_fn(best))

