import numpy as np
import math
import copy
import cma
import sys
import os

from src.gym_TS.agents.TinyAgent import TinyAgent
from src.gym_TS.fitness_calculator import FitnessCalculator
from functools import partial


LOG_EVERY = 20

def rwg(seed_value, calculator, population_size, team_type, target_fitness=1.0):
    """
    Finds a genome with non-zero fitness score by randomly guessing neural network weights. Exists as a helper for CMA
    """
    # RWG does not distinguish between populations and generations
    max_ninds = population_size
    full_genome = None
    backup_genome = None
    max_fitness = -float('Inf')

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

        #if nind%50 == 0:
        #    print(f"{nind}: Best score is {max_fitness}")

        if fitness > max_fitness:
            max_fitness = fitness
            backup_genome = full_genome

        seed_value += 1

    print(f"Did not find a genome with score greater than {target_fitness}. Using best one found, with score {max_fitness}")

    if backup_genome is None:
        return full_genome, max_fitness
    else:
        return backup_genome, max_fitness


def cma_es(fitness_calculator, seed_value, sigma, model_name, results_file_name, team_type, selection_level, num_generations, population_size):
    """
    Evolves a model or pair of models to accomplish the task

    :param fitness_calculator:
    :param seed_value:
    :param sigma:
    :param model_name:
    :param results_file_name:
    :param team_type:
    :param selection_level:
    :param num_generations:
    :param population_size:
    :return:
    """

    options = {'seed': seed_value, 'maxiter': num_generations, 'popsize': population_size, 'tolx': 1e-3, 'tolfunhist': 2e2}

    model_params = model_name.split("_")

    simulation_length = model_params[2]
    num_trials = model_params[4]
    random_seed = model_params[5]
    num_robots = model_params[6]
    num_resources = model_params[7]
    sensor_range = model_params[8]
    slope_angle = model_params[9]
    arena_length = model_params[10]
    arena_width = model_params[11]
    cache_start = model_params[12]
    slope_start = model_params[13]
    source_start = model_params[14]
    upward_cost_factor = model_params[15]
    downward_cost_factor = model_params[16]
    carry_factor = model_params[17]
    resource_reward_factor = model_params[18]

    seed_file = f"bootstrap_{team_type}_{simulation_length}_{num_trials}_{random_seed}_{num_robots}_{num_resources}_{sensor_range}_{slope_angle}_{arena_length}_{arena_width}_{cache_start}_{slope_start}_{source_start}_{upward_cost_factor}_{downward_cost_factor}_{carry_factor}_{resource_reward_factor}.npy"
    seed_genome = None
    # f"CMA_{team_type}_{simulation_length}_{num_generations}_{num_trials}_{random_seed}_{num_robots}_{num_resources}_{sensor_range}_{slope_angle}_{arena_length}_{arena_width}_{cache_start}_{slope_start}_{source_start}_{upward_cost_factor}_{downward_cost_factor}_{carry_factor}_{resource_reward_factor}_{sigma}_{population}"

    try:
        seed_genome = np.load(seed_file)
    except:
        raise RuntimeError("No bootstrap model matches this experiment's parameters")

    seed_fitness = fitness_calculator.calculate_fitness(seed_genome, team_type)

    es = cma.CMAEvolutionStrategy(seed_genome, sigma, options)

    log_file_name = model_name + ".log"

    # Send output to log file
    old_stdout = sys.stdout
    log_file = open(log_file_name, "a")
    sys.stdout = log_file

    partial_calculator = partial(fitness_calculator.calculate_fitness_negation, team_type=team_type)
    # es.optimize(partial_calculator)

    while not es.stop():
        solutions = es.ask()
        es.tell(solutions, fitness(solutions))
        iteration_number = es.result.iterations

        if iteration_number % LOG_EVERY == 0:
            # Log results to results file
            results = model_name.replace("_", ",")
            results += f",{log_file_name}, {seed_fitness}, {es.result[1]}\n"
            intermediate_results_file_name = f"{iteration_number}_{results_file_name}"

            if not os.path.exists(intermediate_results_file_name):
                results_file = open(intermediate_results_file_name, 'a')
                results_file.write("Algorithm Name, Team Type, Simulation Length, Num Generations, Num Trials, "
                                   "Random Seed, Num Robots, Num Resources, Sensor Range, Slope Angle, Arena Length, "
                                   "Arena Width, Cache Start, Slope Start, Source Start, Sigma, Population, Log File, "
                                   "Seed Fitness, Evolved Fitness\n")
            else:
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

        es.disp()

    print(f"Best score is {es.result[1]}")

    ''''''
    sys.stdout = old_stdout
    log_file.close()

    # Append results to results file. Create file if it doesn't exist
    results = model_name.replace("_", ",")
    results += f",{log_file_name}, {seed_fitness}, {es.result[1]}\n"
    results_file = open(results_file_name, 'a')
    results_file.write(results)
    results_file.close()

    return es.result[0]
