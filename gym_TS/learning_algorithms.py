import numpy as np
import math
import copy
import cma

from agents.DQNAgent import DQNAgent
from agents.BasicQAgent import BasicQAgent

from agents.TinyAgent import TinyAgent
from fitness_calculator import FitnessCalculator


# Things to change:
# Mutation method
def genetic_algorithm(calculator, seed_value, output_selection_method, num_generations=2000, population_size=100,
                      num_trials=3, mutation_rate=0.01,
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
        individual = TinyAgent(calculator.get_observation_size(), calculator.get_action_size(), output_selection_method,
                               seed_value)
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
            child = TinyAgent(calculator.get_observation_size(), calculator.get_action_size(), output_selection_method,
                              seed_value)
            child.load_weights(child_genome)
            new_population += [child]

        # Reset values for population and fitness scores
        population = copy.deepcopy(new_population)
        fitness_scores = [-math.inf] * population_size

    return best_individual


def grammatical_evolution(random_seed, num_generations=2000, population_size=100,
                          num_trials=3, mutation_rate=0.01,
                          elitism_percentage=0.05):
    # Run PonyGE
    # python3 ponyge.py --parameters task_specialisation.txt --random_seed random_seed
    pass


def q_learning(calculator, num_episodes, random_seed, batch_size):
    # Information for BasicQ
    front = [[0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0]]
    locations = [[0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0]]
    has_obj = [[0], [1]]

    possible_observations = []

    for x in front:
        for y in locations:
            for z in has_obj:
                possible_observations += [np.array([x + y + z])]

    # agent = BasicQAgent(possible_observations, calculator.get_action_size(), random_seed, batch_size=batch_size)

    agent = DQNAgent(calculator.get_observation_size(), calculator.get_action_size(), random_seed,
                     batch_size=128)  # calculator.simulation_length*calculator.env.num_robots)

    for e in range(num_episodes):
        render = False
        # if e%10 == 0:
        #    render = True
        score, agent = calculator.calculate_fitness(agent, num_trials=5, render=render, learning_method="DQN")
        print(f'Score at episode {e} is {score}')

        if e % 100 == 99:
            agent.save(f"Q_best_{e}")
            agent.display()

    return agent


def rwg(seed_value, calculator, population_size):
    """
    Finds a non-zero value of the fitness function by randomly guessing neural network weights
    """
    # RWG does not distinguish between populations and generations
    max_ninds = population_size

    best_individual = None
    max_score = -math.inf

    # Neuroevolution loop
    for nind in range(max_ninds):

        # Create individual
        individual = TinyAgent(calculator.get_observation_size(), calculator.get_action_size(), seed=seed_value)
        individual.load_weights()  # No parameters means random weights are generated

        # Evaluate individual's fitness
        score = calculator.calculate_fitness(individual, render=False)
        # print(f"{nind} Score: {score}")
        if score > 0.0:
            print(f"Found an individual with non-zero score after {nind} tries")
            return individual

        # Save the best individual
        if score > max_score:
            max_score = score
            best_individual = individual
            if nind != 0:
                calculator.calculate_fitness(best_individual)

        seed_value += 1

    print(f"Did not find an individual with non-zero score. Using first individual")
    return best_individual


def cma_es(fitness_calculator, seed_value, sigma, model_name):
    options = {'seed': seed_value}

    seed_individual = rwg(seed_value=seed_value, calculator=fitness_calculator, population_size=1000)
    seed_weights = seed_individual.get_weights()
    es = cma.CMAEvolutionStrategy(seed_weights, sigma, options)

    #es.optimize(fitness_calculator.calculate_fitness_negation)

    while not es.stop():
        solutions = es.ask()
        es.tell(solutions, [fitness_calculator.calculate_fitness_negation(x) for x in solutions])
        es.logger.add()  # write data to disc to be plotted
        es.disp()

    #es.result_pretty()
    # cma.savefig("Some_figure.png")
    #cma.plot()
    es.logger.save_to(model_name)

    print(f"Best score is {es.result[1]}")
    return es.result[0]
