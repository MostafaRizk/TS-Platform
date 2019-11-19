"""
Modified from code by Giuseppe Cuccu
"""

import numpy as np
import math
import copy
import cma
import sys
import getopt

from agents.TinyAgent import TinyAgent
from fitness_calculator import FitnessCalculator


def rwg(seed_value, population_size=1000):
    # RWG does not distinguish between populations and generations
    max_ninds = population_size

    best_individual = None
    max_score = -math.inf

    # Neuroevolution loop
    for nind in range(max_ninds):

        # Create individual
        individual = TinyAgent(fitness_calculator.get_observation_size(), fitness_calculator.get_action_size(), seed_value)
        individual.load_weights()  # No parameters means random weights are generated

        # Evaluate individual's fitness
        score = fitness_calculator.calculate_fitness(individual)
        print(f"{nind} Score: {score}")

        # Save the best individual
        if score > max_score:
            max_score = score
            best_individual = individual
            if nind != 0:
                fitness_calculator.calculate_fitness(best_individual)
                # best_individual.save_model()

        seed_value += 1

    return best_individual


# Things to change:
# Mutation method
def genetic_algorithm(calculator, seed_value, output_selection_method, num_generations=2000, population_size=100, num_trials=3, mutation_rate=0.01,
                      elitism_percentage=0.05):
    # population_size = 10
    # num_generations = 40
    # num_trials = 3
    # mutation_rate = 0.01
    crossover_rate = 0.3  # Single-point crossover
    # elitism_percentage = 0.05  # Generational replacement with roulette-wheel selection
    num_elite = max(2, int(population_size * elitism_percentage))

    model_saving_rate = 10

    # Create randomly initialised population
    population = []

    for i in range(population_size):
        individual = TinyAgent(calculator.get_observation_size(), calculator.get_action_size(), output_selection_method, seed_value)
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
                avg_fitness += calculator.calculate_fitness(population[i])
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

        best_individual = new_population[0]

        if generation % model_saving_rate == 0:
            print(f"Best score at generation {generation} is {best_score} ")
            best_individual.save_model(f"{output_selection_method}_{generation}")

        # Create new generation
        for i in range(population_size - num_elite):
            # Choose parents
            parent1_genome = new_population[0].get_weights()
            parent2_genome = new_population[1].get_weights()

            # Do crossover
            rng = calculator.get_rng()
            crossover_point = rng.randint(low=0, high=len(parent1_genome))
            child_genome = np.append(parent1_genome[0:crossover_point], parent2_genome[crossover_point:])

            # Mutate with probability and add to population
            # For every gene in the genome (i.e. weight in the neural network)
            for j in range(len(child_genome)):
                decimal_places = 16
                gene = child_genome[j]

                # Create a bit string from the float of the genome
                bit_string = bin(int(gene * (10 ** decimal_places)))  # Gene is a float, turn into int
                start_index = bit_string.find('b') + 1  # Prefix is either 0b or -0b. Start after 'b'
                bit_list = list(bit_string[start_index:])

                # Iterate through bit string and flip each bit according to mutation probability
                for b in range(len(bit_list)):
                    random_number = rng.rand()

                    if random_number < mutation_rate:
                        if bit_list[b] == '0':
                            bit_list[b] = '1'
                        else:
                            bit_list[b] = '0'

                mutated_gene = float(int(''.join(bit_list), 2)) / (10 ** decimal_places)
                child_genome[j] = mutated_gene

            # Create child individual and add to population
            child = TinyAgent(calculator.get_observation_size(), calculator.get_action_size(), output_selection_method, seed_value)
            child.load_weights(child_genome)
            new_population += [child]

        # Reset values for population and fitness scores
        population = copy.deepcopy(new_population)
        fitness_scores = [-math.inf] * population_size

    return best_individual


def cma_es(calculator, seed_value, sigma=0.5):
    demo_agent = TinyAgent(calculator.get_observation_size(), calculator.get_action_size(), seed_value)
    num_weights = demo_agent.get_num_weights()
    # res = cma.fmin(fitness, num_weights * [0], 0.5)
    es = cma.CMAEvolutionStrategy(num_weights * [0], sigma).optimize(calculator.calculate_fitness)
    print(f"Best score is {es.result[1]}")
    return es.result[0]


def grammatical_evolution():
    # Run PonyGE
    # python3 ponyge.py --parameters task_specialisation.txt --random_seed random_seed
    pass


# Replay winning individual
def evaluate_best(calculator, best, seed, num_trials=100):
    if best:
        test_scores = []
        avg_score = 0

        for i in range(num_trials):
            render_flag = False
            if i == 0:
                render_flag = True
            test_scores += [calculator.calculate_fitness(best, render=render_flag)]
            seed += 1

        avg_score = sum(test_scores) / len(test_scores)
        print(f"The best individual scored {avg_score} on average")


# best_individual = rwg(random_seed)
#best_individual = genetic_algorithm(random_seed, num_generations=100, population_size=5, num_trials=3)

# CMA segment
#best_genome = cma_es(0.5)
#best_individual = TinyAgent(fitness_calculator.get_observation_size(), fitness_calculator.get_action_size(), random_seed)
#best_individual.load_weights(best_genome)

# Testing & saving segment
#evaluate_best(best_individual, random_seed)
#best_individual.save_model()

def main(argv):

    # Necessary arguments
    # Algorithm ?
    # Generations
    # Population
    # Trials
    # argmax vs probability

    try:
        opts, args = getopt.getopt(argv, "", ["algorithm=", "generations=", "population=", "trials=", "simulation_length=", "seed=", "argmax="])

    except getopt.GetoptError:
        print("There was an error")
        sys.exit(2)

    training_algorithm = None
    num_generations = None
    population_size = None
    num_trials = None
    simulation_length = None
    selection_method = None
    random_seed = None

    for opt, arg in opts:
        if opt == "--algorithm":
            if arg == "ga":
                training_algorithm = genetic_algorithm
            else:
                print("This algorithm is either misspelled or not supported")
                sys.exit(2)
        if opt == "--generations":
            num_generations = int(arg)
        if opt == "--population":
            population_size = int(arg)
        if opt == "--trials":
            num_trials = int(arg)
        if opt == "--simulation_length":
            simulation_length = int(arg)
        if opt == "--argmax":
            if arg == "True":
                selection_method = "argmax"
            elif arg == "False":
                selection_method = "weighted_probability"
            else:
                raise IOError("Please use 'True' or 'False' for argmax")
        if opt == "--seed":
            random_seed = int(arg)

    fitness_calculator = FitnessCalculator(random_seed=random_seed,
                                           simulation_length=simulation_length,
                                           output_selection_method=selection_method)

    best_individual = training_algorithm(calculator=fitness_calculator,
                                         seed_value=random_seed,
                                         output_selection_method=selection_method,
                                         num_generations=num_generations,
                                         population_size=population_size,
                                         num_trials=num_trials)
    best_individual.save_model(f"{selection_method}_final")

    #evaluate_best(calculator=fitness_calculator, seed=random_seed, best=best_individual, num_trials=2)


# To run, use:
# python3 TS_NE.py --algorithm ga --generations 5 --population 5 --trials 3 --simulation_length 1000 --argmax True --seed 1
if __name__ == "__main__":
    #main(sys.argv[1:])
    fitness_calculator = FitnessCalculator(random_seed=1,
                                           simulation_length=1000,
                                           output_selection_method="argmax")
    best_individual = TinyAgent(fitness_calculator.get_observation_size(), fitness_calculator.get_action_size(),
                                1)
    best_genome = cma_es(calculator = fitness_calculator, seed_value=1, sigma=0.5)
    best_individual.load_weights(best_genome)
    #evaluate_best(calculator=fitness_calculator, seed=1, best=best_individual, num_trials=2)
