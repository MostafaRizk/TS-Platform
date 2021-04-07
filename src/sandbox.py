import json
import numpy as np
import time

from qdpy.algorithms import *
from qdpy.containers import *
from qdpy.benchmarks import *
from qdpy.plots import *
from qdpy.base import *
from qdpy import tools

from fitness import FitnessCalculator
from agents.nn_agent_lean import NNAgent
from operator import add
from learning.cma_centralised import CentralisedCMALearner

param_file = "default_parameters.json"
parameter_dictionary = json.loads(open(param_file).read())
num_agents = 2
num_actions = 6
episode_length = 100
num_generations = 50
pop_size = 200
fitness_calculator = FitnessCalculator(param_file)
genome_length = NNAgent(fitness_calculator.get_observation_size(), fitness_calculator.get_action_size(), param_file).get_num_weights() * num_agents


def eval_fn(full_genome, bc_measure="total_action_count"):
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

    if bc_measure == "avg_y" or bc_measure == "total_action_count":
        features = tuple(np.mean(results['behaviour_characterisation_list'], axis=0))

    elif bc_measure == "action_count":
        bc = np.mean(results['behaviour_characterisation_list'], axis=0)
        bc = bc.flatten()
        features = tuple(bc)

    return (team_score,), features


#grid_shape = (8, 8)
#bc_range = ((0, 7), (0, 7))

#grid_shape = tuple([5]*(num_actions*num_agents))
#bc_range = tuple([(0, num_episodes) for i in range(num_actions*num_agents)])

grid_shape = tuple([2]*num_actions)
bc_range = tuple([(0, episode_length) for i in range(num_actions)])

grid = containers.Grid(shape=grid_shape, max_items_per_bin=1, fitness_domain=((-100000, 100000),), features_domain=bc_range)
options = {'seed': parameter_dictionary['general']['seed'],
           #'maxiter': parameter_dictionary['algorithm']['cma']['generations'],
           'maxiter': num_generations,
           #'popsize': self.get_genome_population_length(),
           'popsize': pop_size,
           'tolx': parameter_dictionary['algorithm']['cma']['tolx'],
           'tolfunhist': parameter_dictionary['algorithm']['cma']['tolfunhist'],
           'tolflatfitness': parameter_dictionary['algorithm']['cma']['tolflatfitness'],
           'tolfun': parameter_dictionary['algorithm']['cma']['tolfun']}

#algo = algorithms.CMAES(grid, budget=num_generations, dimension=genome_length, sigma0=0.2)
algo = algorithms.CMAMEImprovementEmitter(grid, budget=num_generations, dimension=genome_length, sigma0=0.2, optimisation_task="maximisation")

"""
emitters = [algorithms.CMAMEImprovementEmitter(grid, budget=num_generations, dimension=genome_length, sigma0=0.2, optimisation_task="maximisation"),
            algorithms.CMAMEImprovementEmitter(grid, budget=num_generations, dimension=genome_length, sigma0=0.2, optimisation_task="maximisation"),
            algorithms.CMAMEImprovementEmitter(grid, budget=num_generations, dimension=genome_length, sigma0=0.2, optimisation_task="maximisation")]
algo = algorithms.AlternatingAlgWrapper(algorithms=emitters)
"""

start = time.perf_counter()
best = algo.optimise(eval_fn, batch_mode=False)
end = time.perf_counter()

results = eval_fn(best)
print(results)
print(f"Time taken: {end-start}")
model_prefix = CentralisedCMALearner.get_model_name_from_dictionary(parameter_dictionary, results[0][0])
NNAgent.save_given_model(best, f"{model_prefix}.npy")

#((164.00000000000003,), (0.0, 5.287999999999996)) #10 generations default pop
#((-1412.32,), (0.06, 2.9919999999999978)) #100 generations default pop
#((-497.12000000000006,), (2.777999999999997, 0.4620000000000003)) # 100 pop, 1000 gen, sigma 0.8
