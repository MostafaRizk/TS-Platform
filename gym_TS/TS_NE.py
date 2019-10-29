"""
Modified from code by Giuseppe Cuccu
"""

import numpy as np
import math
import gym
import copy
import cma

from gym_TS.agents.TinyAgent import TinyAgent
from gym.utils import seeding
from sklearn.preprocessing import OneHotEncoder

env = gym.make('gym_TS:TS-v2')
# env = gym.wrappers.Monitor(env, 'video', force = True) # Uncomment to save video

# categories = env.get_observation_categories()
# one_hot_encoder = OneHotEncoder(sparse=False, categories=categories)

# Get size of input and output space and creates agent
# observation_size = env.get_observation_size() * len(env.get_observation_categories()[0])  # env.get_observation_size()
observation_size = env.get_observation_size()
action_size = env.get_action_size()

# random_seed = np.random.randint(1e10)
random_seed = 0

# Action space uses a separate random number generator so need to set its seed separately
env.action_space.np_random.seed(random_seed)

np_random, seed = seeding.np_random(random_seed)


# Fitness function: gameplay loop
def fitness(individual, render=False):
    # env.seed(random_seed)  # makes fitness deterministic
    env.seed(random_seed)
    observations = env.reset()
    score = 0
    done = False

    for t in range(simulation_length):
        if render:
            env.render()

        # observation = np.reshape(observation, [1, env.get_observation_size()])
        # encoded_observations = [one_hot_encoder.fit_transform(observation.reshape(1, -1)) for observation in observations]

        # All agents act using same controller.
        # robot_actions = [individual.act(np.reshape(observations[j], [1, env.get_observation_size()]))
        #                 for j in range(env.get_num_robots())]
        # robot_actions = [individual.act(encoded_observations[i]) for i in range(env.get_num_robots())]

        robot_actions = [env.action_space.sample() for el in range(env.get_num_robots())]

        # The environment changes according to all their actions
        observations, reward, done, info = env.step(robot_actions, t)
        score += reward

        if done:
            break

    return score


simulation_length = 1000


def rwg(seed_value, population_size=1000):
    # RWG does not distinguish between populations and generations
    max_ninds = population_size

    best_individual = None
    max_score = -math.inf

    # Neuroevolution loop
    for nind in range(max_ninds):

        # Create individual
        individual = TinyAgent(observation_size, action_size, random_seed)
        individual.load_weights()  # No parameters means random weights are generated

        # Evaluate individual's fitness
        score = fitness(individual)
        print(f"{nind} Score: {score}")

        # Save the best individual
        if score > max_score:
            max_score = score
            best_individual = individual
            if nind != 0:
                fitness(best_individual)
                # best_individual.save_model()

        seed_value += 1

    return best_individual


# Things to change:
# Mutation method
def ga(seed_value, num_generations=2000, population_size=100, mutation_rate=0.01, elitism_percentage=0.05):
    # population_size = 10
    # num_generations = 40
    num_trials = 3
    # mutation_rate = 0.01
    crossover_rate = 0.3  # Single-point crossover
    # elitism_percentage = 0.05  # Generational replacement with roulette-wheel selection
    num_elite = max(2, int(population_size * elitism_percentage))

    # Create randomly initialised population
    population = []

    for i in range(population_size):
        individual = TinyAgent(observation_size, action_size, seed_value)
        individual.load_weights()  # No parameters means random weights are generated
        population += [individual]
        seed_value += 1

    fitness_scores = []
    best_individual = None

    for generation in range(num_generations):

        for i in range(population_size):
            # Get fitness of individual and add to fitness_scores
            avg_fitness = 0
            for trial in range(num_trials):
                avg_fitness += fitness(population[i])
            avg_fitness /= num_trials
            fitness_scores += [avg_fitness]

        new_population = []

        # Add elite to new_population
        for e in range(num_elite):
            best_score = -math.inf
            best_index = 0
            for ind in range(population_size):
                if fitness_scores[ind] > best_score:
                    best_score = fitness_scores[ind]
                    best_index = ind
            new_population += [copy.deepcopy(population[best_index])]
            fitness_scores[best_index] = -math.inf

            if e == 0:
                print(f"Best score at generation {generation} is {best_score} ")

        best_individual = new_population[0]

        # Create new generation
        for i in range(population_size - num_elite):
            # Choose parents
            parent1_genome = new_population[0].get_weights()
            parent2_genome = new_population[1].get_weights()

            # Do crossover
            crossover_point = np_random.randint(low=0, high=len(parent1_genome))
            child_genome = np.append(parent1_genome[0:crossover_point], parent2_genome[crossover_point:])

            # Mutate with probability and add to population
            for j in range(len(child_genome)):
                random_number = np_random.rand()
                if random_number < mutation_rate:
                    child_genome[j] = np_random.rand()

            # Create child individual and add to population
            child = TinyAgent(observation_size, action_size, seed_value)
            child.load_weights(child_genome)
            new_population += [child]

        population = copy.deepcopy(new_population)

    return best_individual


def cma_es():
    es = cma.CMAEvolutionStrategy(60 * [0], 0.5)
    while not es.stop():
        solutions = es.ask()
        population = []
        for x in solutions:
            individual = TinyAgent(observation_size, action_size, random_seed)
            individual.load_weights(np.array(x))
            population += [individual]
        #es.tell(solutions, [fitness(individual) for individual in population])
        #es.logger.add()  # write data to disc to be plotted
        #es.disp()
        #es.result_pretty()
        #cma.plot()  # shortcut for es.logger.plot()

# Replay winning individual
def evaluate_best(best, seed):
    if best:
        test_scores = []
        avg_score = 0
        for i in range(100):
            render_flag = False
            if i == 0:
                render_flag = True
            test_scores += [fitness(best, render=render_flag)]
            seed += 1
        avg_score = sum(test_scores) / len(test_scores)
        print(f"The best individual scored {avg_score} on average")


# best_individual = rwg(random_seed, 10)
#best_individual = ga(random_seed, num_generations=10, population_size=20)

#evaluate_best(best_individual, random_seed)
cma_es()