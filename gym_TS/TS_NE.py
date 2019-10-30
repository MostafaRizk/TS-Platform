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
# from sklearn.preprocessing import OneHotEncoder

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
def ga(seed_value, num_generations=2000, population_size=100, num_trials=3, mutation_rate=0.01, elitism_percentage=0.05):
    # population_size = 10
    # num_generations = 40
    # num_trials = 3
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

    fitness_scores = [-math.inf] * population_size
    best_individual = None

    for generation in range(num_generations):

        for i in range(population_size):
            # Get fitness of individual and add to fitness_scores
            avg_fitness = 0
            for trial in range(num_trials):
                avg_fitness += fitness(population[i])
            avg_fitness /= num_trials
            fitness_scores[i] = avg_fitness

        new_population = []

        # Add elite to new_population
        for e in range(num_elite):
            best_score = -math.inf
            best_index = 0

            # Find best individual (excluding last selected elite)
            for ind in range(population_size):
                if fitness_scores[ind] > best_score:
                    best_score = fitness_scores[ind]
                    best_index = ind

            # Add best individual to new population and exclude them from next iteration
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
            # For every gene in the genome (i.e. weight in the neural network)
            for j in range(len(child_genome)):
                decimal_places = 16
                gene = child_genome[j]

                # Create a bit string from the float of the genome
                bit_string = bin(int(gene*(10**decimal_places)))  # Gene is a float, turn into int
                start_index = bit_string.find('b') + 1  # Prefix is either 0b or -0b. Start after 'b'
                bit_list = list(bit_string[start_index:])

                # Iterate through bit string and flip each bit according to mutation probability
                for b in range(len(bit_list)):
                    random_number = np_random.rand()

                    if random_number < mutation_rate:
                        if bit_list[b] == '0':
                            bit_list[b] = '1'
                        else:
                            bit_list[b] = '0'

                mutated_gene = float(int(''.join(bit_list), 2)) / (10**decimal_places)
                child_genome[j] = mutated_gene

            # Create child individual and add to population
            child = TinyAgent(observation_size, action_size, seed_value)
            child.load_weights(child_genome)
            new_population += [child]

        # Reset values for population and fitness scores
        population = copy.deepcopy(new_population)
        fitness_scores = [-math.inf] * population_size

    return best_individual


def cma_es():
    res = cma.fmin(fitness, 60 * [0], 0.5)
    es = cma.CMAEvolutionStrategy(60 * [0], 0.5).optimize(fitness)
    print(f"Best score iss {es.result[1]}")
    return es.result[0]

# Replay winning individual
def evaluate_best(best, seed, num_trials=100):
    if best:
        test_scores = []
        avg_score = 0
        for i in range(num_trials):
            render_flag = False
            if i == 0:
                render_flag = True
            test_scores += [fitness(best, render=render_flag)]
            seed += 1
        avg_score = sum(test_scores) / len(test_scores)
        print(f"The best individual scored {avg_score} on average")


#best_individual = rwg(random_seed, 100)
#best_individual = ga(random_seed, num_generations=20, population_size=5, num_trials=5)
best_genome = cma_es()
best_individual = TinyAgent(observation_size, action_size, random_seed)
best_individual.load_weights(best_genome)

evaluate_best(best_individual, random_seed)

