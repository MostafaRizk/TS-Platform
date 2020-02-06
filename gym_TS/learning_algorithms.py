import numpy as np
import math
import copy
import cma
import sys
import os

from agents.DQNAgent import DQNAgent
from agents.BasicQAgent import BasicQAgent

from agents.TinyAgent import TinyAgent
from fitness_calculator import FitnessCalculator
from functools import partial


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


def rwg(seed_value, calculator, population_size, team_type, target_fitness=1.0):
    """
    Finds a genome with non-zero fitness score by randomly guessing neural network weights. Exists as a helper for CMA
    """
    # RWG does not distinguish between populations and generations
    max_ninds = population_size
    full_genome = None
    backup_genome = None
    max_fitness = 0.0

    # Neuroevolution loop
    for nind in range(max_ninds):

        if team_type == "homogeneous":
            # Create individual
            individual = TinyAgent(calculator.get_observation_size(), calculator.get_action_size(), seed=seed_value)
            individual.load_weights()  # No parameters means random weights are generated
            full_genome = individual.get_weights()
        elif team_type == "heterogeneous":
            individual1 = TinyAgent(calculator.get_observation_size(), calculator.get_action_size(), seed=seed_value)
            individual2 = TinyAgent(calculator.get_observation_size(), calculator.get_action_size(), seed=seed_value)
            individual1.load_weights()  # No parameters means random weights are generated
            individual2.load_weights()
            full_genome = np.concatenate([individual1.get_weights(), individual2.get_weights()])
        else:
            raise RuntimeError("Did not use a valid team type")

        # Evaluate individual's fitness
        fitness = calculator.calculate_fitness(full_genome, team_type=team_type, render=False)
        # print(f"{nind} Score: {score}")

        if fitness >= target_fitness:
            print(f"Found an individual with score {fitness} >= {target_fitness} after {nind} tries")
            return full_genome, fitness
        elif fitness > 0.0:
            print(f"Found an individual with score {fitness} > 0 after {nind} tries")
        #elif nind%10 == 0:
        #    print(f"{nind}: Best score is {max_fitness}")

        if fitness > max_fitness:
            max_fitness = fitness
            backup_genome = full_genome

        seed_value += 1

    print(f"Did not find a genome with score greater than 2. Using best one found, with score {max_fitness}")


    if backup_genome is None:
        return full_genome, max_fitness
    else:
        return backup_genome, max_fitness


def cma_es(fitness_calculator, seed_value, sigma, model_name, results_file_name, team_type):
    options = {'seed': seed_value, 'maxiter': 300, 'popsize': 40}

    #seed_genome = rwg(seed_value=seed_value, calculator=fitness_calculator, population_size=2000, team_type=team_type)
    #f"bootstrap_{team_type}_{simulation_length}_{num_generations}_{num_trials}_{random_seed}_{num_robots}_{num_resources}_{sensor_range}_{slope_angle}_{arena_length}_{arena_width}_{cache_start}_{slope_start}_{source_start}_{best_fitness}"
    #f"CMA_{team_type}_{simulation_length}_{num_generations}_{num_trials}_{random_seed}_{num_robots}_{num_resources}_{sensor_range}_{slope_angle}_{arena_length}_{arena_width}_{cache_start}_{slope_start}_{source_start}_{sigma}"


    # Get available bootstrapped models and sort them
    bootstrap_directory = "models/bootstrap/"
    available_files = os.listdir(bootstrap_directory)
    seed_file = None

    # Make sure there are files
    if available_files == []:
       raise RuntimeError("No bootstrapped models available")

    # Sort the files and generate a list of random indices to allow random selection of one of the models
    available_files.sort()
    rng = fitness_calculator.get_rng()
    index_list = list(range(0, len(available_files)))
    rng.shuffle(index_list)

    # Choose a random model from those available
    for i in range(len(index_list)):
        parameter_list = available_files[i].split("_")

        # Check that important experiment parameters are the same (team type, simulation length, num robots, num resources, sensor range, slope angle and arena measurements)
        if parameter_list[1] == model_name.split("_")[1] and parameter_list[2] == model_name.split("_")[2] and parameter_list[6:15] == model_name.split("_")[6:15]:
            seed_file = bootstrap_directory + available_files[i]
            break

    if seed_file is None:
        raise RuntimeError("No bootstrap model matches this experiment's parameters")

    seed_genome = np.load(seed_file)
    es = cma.CMAEvolutionStrategy(seed_genome, sigma, options)

    '''
    # Send output to log file
    old_stdout = sys.stdout
    log_file_name = model_name + ".log"
    log_file = open(log_file_name, "a")
    sys.stdout = log_file
    '''


    partial_calculator = partial(fitness_calculator.calculate_fitness_negation, team_type=team_type)
    #es.optimize(partial_calculator)

    while not es.stop():
        solutions = es.ask()
        es.tell(solutions, [partial_calculator(x) for x in solutions])
        iteration_number = es.result.iterations

        if iteration_number % 20 == 0:
            # Log results to results file
            results = model_name.replace("_", ",")
            results += f",{log_file_name}, {es.result[1]}\n"
            intermediate_results_file_name = f"{iteration_number}_{results_file_name}"
            results_file = open(intermediate_results_file_name, 'a')
            results_file.write(results)
            results_file.close()

            # Log genome
            if team_type == "homogeneous":
                best_individual = TinyAgent(fitness_calculator.get_observation_size(),
                                            fitness_calculator.get_action_size(),
                                            seed=seed_value)
                best_individual.load_weights(es.result[0])
                best_individual.save_model(model_name, sub_dir=str(iteration_number))

            # Split the genome and save both halves separately for heterogeneous setup
            elif team_type == "heterogeneous":
                best_individual_1 = TinyAgent(fitness_calculator.get_observation_size(),
                                              fitness_calculator.get_action_size(),
                                              seed=seed_value)
                best_individual_2 = TinyAgent(fitness_calculator.get_observation_size(),
                                              fitness_calculator.get_action_size(),
                                              seed=seed_value)

                # Split genome
                mid = int(len(es.result[0]) / 2)
                best_individual_1.load_weights(es.result[0][0:mid])
                best_individual_2.load_weights(es.result[0][mid:])

                best_individual_1.save_model(model_name + "_controller1_", sub_dir=str(iteration_number))
                best_individual_2.save_model(model_name + "_controller2_", sub_dir=str(iteration_number))

    #    es.logger.add()  # write data to disc to be plotted
        es.disp()

    #es.result_pretty()
    # cma.savefig("Some_figure.png")
    #cma.plot()
    #es.logger.save_to(model_name)

    print(f"Best score is {es.result[1]}")

    '''
    sys.stdout = old_stdout
    log_file.close()
    '''


    # Append results to results file. Create file if it doesn't exist
    results = model_name.replace("_", ",")
    results += f",{log_file_name}, {es.result[1]}\n"
    results_file = open(results_file_name, 'a')
    results_file.write(results)
    results_file.close()

    return es.result[0]
