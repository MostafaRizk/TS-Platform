import sys
import getopt
import os
import numpy as np

from agents.DQNAgent import DQNAgent
from agents.BasicQAgent import BasicQAgent

from agents.TinyAgent import TinyAgent
from fitness_calculator import FitnessCalculator

from learning_algorithms import cma_es
from learning_algorithms import rwg


def main(argv):
    try:
        opts, args = getopt.getopt(argv, "",
                                   ["algorithm=", "team_type=", "generations=", "simulation_length=", "trials=",
                                    "seed=", "num_robots=", "num_resources=", "sensor_range=", "slope_angle=",
                                    "arena_length=", "arena_width=", "cache_start=", "slope_start=", "source_start=",
                                    "sigma=", "test_model=", "target_fitness=", "population="])

    except getopt.GetoptError:
        print("There was an error")
        sys.exit(2)

    bootstrap = False

    training_algorithm = None
    team_type = None
    simulation_length = None
    num_generations = None
    num_trials = None
    random_seed = None

    num_robots = None
    num_resources = None
    sensor_range = None
    slope_angle = None
    arena_length = None
    arena_width = None
    cache_start = None
    slope_start = None
    source_start = None
    sigma = None
    test_model = None
    target_fitness = 1.0
    population = None

    # Read in arguments
    for opt, arg in opts:
        if opt == "--algorithm":
            if arg == "cma":
                training_algorithm = cma_es
            elif arg == "bootstrap":
                training_algorithm = rwg
                bootstrap = True
            else:
                print("This algorithm is either misspelled or not supported yet")
                sys.exit(2)
        if opt == "--team_type":
            if arg == "homogeneous" or arg == "heterogeneous":
                team_type = arg
            else:
                print("Team type is misspelled or unsupported. Use \"homogeneous\" or \"heterogeneous\"")
        if opt == "--generations":
            num_generations = int(arg)
        if opt == "--simulation_length":
            simulation_length = int(arg)
        if opt == "--trials":
            num_trials = int(arg)
        if opt == "--seed":
            random_seed = int(arg)
        if opt == "--num_robots":
            num_robots = int(arg)
        if opt == "--num_resources":
            num_resources = int(arg)
        if opt == "--sensor_range":
            sensor_range = int(arg)
        if opt == "--slope_angle":
            slope_angle = int(arg)
        if opt == "--arena_length":
            arena_length = int(arg)
        if opt == "--arena_width":
            arena_width = int(arg)
        if opt == "--cache_start":
            cache_start = int(arg)
        if opt == "--slope_start":
            slope_start = int(arg)
        if opt == "--source_start":
            source_start = int(arg)
        if opt == "--sigma":
            sigma = float(arg)
        if opt == "--test_model":
            test_model = arg
        if opt == "--target_fitness":
            target_fitness = float(arg)
        if opt == "--population":
            population = int(arg)

    if bootstrap:
        fitness_calculator = FitnessCalculator(random_seed=random_seed, simulation_length=simulation_length,
                                               num_trials=num_trials, num_robots=num_robots,
                                               num_resources=num_resources,
                                               sensor_range=sensor_range, slope_angle=slope_angle,
                                               arena_length=arena_length, arena_width=arena_width,
                                               cache_start=cache_start,
                                               slope_start=slope_start, source_start=source_start)

        best_genome, best_fitness = training_algorithm(seed_value=random_seed, calculator=fitness_calculator, population_size=num_generations, team_type=team_type, target_fitness=target_fitness)

        model_name = f"bootstrap_{team_type}_{simulation_length}_{num_generations}_{num_trials}_{random_seed}_{num_robots}_{num_resources}_{sensor_range}_{slope_angle}_{arena_length}_{arena_width}_{cache_start}_{slope_start}_{source_start}_{best_fitness}"

        directory = "models/bootstrap/"

        if not os.path.exists(directory):
            os.makedirs(directory)

        np.save(directory+model_name, best_genome)

    else:
        # If this is a training run
        if test_model is None:

            # Prepare fitness function
            fitness_calculator = FitnessCalculator(random_seed=random_seed, simulation_length=simulation_length,
                                                   num_trials=num_trials, num_robots=num_robots,
                                                   num_resources=num_resources,
                                                   sensor_range=sensor_range, slope_angle=slope_angle,
                                                   arena_length=arena_length, arena_width=arena_width,
                                                   cache_start=cache_start,
                                                   slope_start=slope_start, source_start=source_start)

            model_name = f"CMA_{team_type}_{simulation_length}_{num_generations}_{num_trials}_{random_seed}_{num_robots}_{num_resources}_{sensor_range}_{slope_angle}_{arena_length}_{arena_width}_{cache_start}_{slope_start}_{source_start}_{sigma}_{population}"

            # Create results file if it doesn't exist
            results_file_name = "results.csv"

            if os.path.exists(results_file_name):
                pass
            else:
                results_file = open(results_file_name, 'w')
                results_file.write("Algorithm Name, Team Type, Simulation Length, Num Trials, Random Seed, Num Robots, Num Resources, Sensor Range, Slope Angle, Arena Length, Arena Width, Cache Start, Slope Start, Source Start, Sigma, Population, Log File, Fitness\n")
                results_file.close()

            print(f"Evaluating {model_name}")

            # Get best genome using CMA
            best_genome = training_algorithm(fitness_calculator=fitness_calculator, seed_value=random_seed, sigma=sigma, model_name=model_name, results_file_name=results_file_name, team_type=team_type, num_generations=num_generations, population_size=population)

            # Create individual using genome so that it can be saved
            if team_type == "homogeneous":
                best_individual = TinyAgent(fitness_calculator.get_observation_size(), fitness_calculator.get_action_size(),
                                            seed=random_seed)
                best_individual.load_weights(best_genome)
                best_individual.save_model(model_name)

            # Split the genome and save both halves separately for heterogeneous setup
            elif team_type == "heterogeneous":
                best_individual_1 = TinyAgent(fitness_calculator.get_observation_size(), fitness_calculator.get_action_size(),
                                            seed=random_seed)
                best_individual_2 = TinyAgent(fitness_calculator.get_observation_size(), fitness_calculator.get_action_size(),
                                            seed=random_seed)

                # Split genome
                mid = int(len(best_genome)/2)
                best_individual_1.load_weights(best_genome[0:mid])
                best_individual_2.load_weights(best_genome[mid:])

                best_individual_1.save_model(model_name + "_controller1_")
                best_individual_2.save_model(model_name + "_controller2_")

        # If this is a testing run
        else:
            model_name = test_model.split("_")
            # CMA_homogeneous_1000_10_10_2_3_1_20_8_4_1_3_7_0.05.npy
            # f"CMA_{team_type}_{simulation_length}_{num_trials}_{random_seed}_{num_robots}_{num_resources}_{sensor_range}_{slope_angle}_{arena_length}_{arena_width}_{cache_start}_{slope_start}_{source_start}_{sigma}"

            #for i in range(len(model_name)):
            #    print(f"{i} - {model_name[i]}")

            # Prepare fitness function
            fitness_calculator = FitnessCalculator(random_seed=int(model_name[5]), simulation_length=int(model_name[3]),
                                                   num_trials=1, num_robots=int(model_name[6]),
                                                   num_resources=int(model_name[7]),
                                                   sensor_range=int(model_name[8]), slope_angle=int(model_name[9]),
                                                   arena_length=int(model_name[10]), arena_width=int(model_name[11]),
                                                   cache_start=int(model_name[12]),
                                                   slope_start=int(model_name[13]), source_start=int(model_name[14]))

            team_type = model_name[2]

            if team_type == "homogeneous":
                full_genome = np.load(test_model)
            elif team_type == "heterogeneous":
                if model_name[16] != "controller1":
                    raise RuntimeError("Use controller 1's path")

                controller1 = np.load(test_model)
                test_model2 = test_model.replace("controller1", "controller2")
                controller2 = np.load(test_model2)
                full_genome = np.concatenate([controller1, controller2])

            fitness_calculator.calculate_fitness(full_genome, team_type=model_name[2], render=False)


# Debugging Tools
def time_fitness_function():
    #team_type = "homogeneous"
    team_type = "heterogeneous"

    fitness_calculator = FitnessCalculator(random_seed=1, simulation_length=1000,
                                           num_trials=1, num_robots=2,
                                           num_resources=3,
                                           sensor_range=1, slope_angle=40,
                                           arena_length=8, arena_width=4,
                                           cache_start=1,
                                           slope_start=3, source_start=7)

    full_genome = None

    if team_type == "homogeneous":
        individual = TinyAgent(fitness_calculator.get_observation_size(), fitness_calculator.get_action_size(), seed=1)
        individual.load_weights()  # No parameters means random weights are generated
        full_genome = individual.get_weights()
    elif team_type == "heterogeneous":
        individual1 = TinyAgent(fitness_calculator.get_observation_size(), fitness_calculator.get_action_size(), seed=1)
        individual2 = TinyAgent(fitness_calculator.get_observation_size(), fitness_calculator.get_action_size(), seed=1)
        individual1.load_weights()  # No parameters means random weights are generated
        individual2.load_weights()
        full_genome = np.concatenate([individual1.get_weights(), individual2.get_weights()])

    import time

    print("It is beginning")

    avg_time_taken = 0
    num_iterations = 30

    for i in range(num_iterations):
        start = time.time()
        score = fitness_calculator.calculate_fitness_negation(full_genome, team_type=team_type, render=False)  # True)#
        end = time.time()

        time_taken = end - start
        avg_time_taken += time_taken

    avg_time_taken /= num_iterations

    print(f"It took {avg_time_taken}")

def time_rwg():
    team_type = "homogeneous"
    #team_type = "heterogeneous"

    fitness_calculator = FitnessCalculator(random_seed=1, simulation_length=100,
                                           num_trials=1, num_robots=2,
                                           num_resources=3,
                                           sensor_range=1, slope_angle=40,
                                           arena_length=8, arena_width=4,
                                           cache_start=1,
                                           slope_start=3, source_start=7)

    import time

    print("It is beginning")

    avg_time_taken = 0
    num_iterations = 1

    for i in range(num_iterations):
        start = time.time()
        seed_genome = rwg(seed_value=1, calculator=fitness_calculator, population_size=5000, team_type=team_type,
                          model_name="debugging")
        end = time.time()

        time_taken = end - start
        avg_time_taken += time_taken

    avg_time_taken /= num_iterations

    print(f"It took {avg_time_taken}")

def time_cma():
    team_type = "homogeneous"
    #team_type = "heterogeneous"

    fitness_calculator = FitnessCalculator(random_seed=1, simulation_length=100,
                                           num_trials=1, num_robots=2,
                                           num_resources=3,
                                           sensor_range=1, slope_angle=40,
                                           arena_length=8, arena_width=4,
                                           cache_start=1,
                                           slope_start=3, source_start=7)

    import time

    print("It is beginning")

    avg_time_taken = 0
    num_iterations = 1

    for i in range(num_iterations):
        start = time.time()
        seed_genome = rwg(seed_value=1, calculator=fitness_calculator, population_size=5000, team_type=team_type,
                          model_name="debugging")
        end = time.time()

        time_taken = end - start
        avg_time_taken += time_taken

    avg_time_taken /= num_iterations

    print(f"It took {avg_time_taken}")

# To run, use:
# python3 main.py --algorithm cma --team_type homogeneous --simulation_length 1000 --trials 3 --seed 100 --num_robots 2 --num_resources 3 --sensor_range 1 --slope_angle 40 --arena_length 8 --arena_width 4 --cache_start 1 --slope_start 3 --source_start 7 --sigma 0.05
# python3 main.py --test_model /home/mriz9/Documents/Results/AAMAS/3_PostGiuse/CMA_homogeneous_1000_10_18_2_3_1_0_8_4_1_3_7_0.05.log
# python3 main.py --algorithm bootstrap --team_type homogeneous --generations 1 --simulation_length 1000 --trials 1 --seed 100 --num_robots 2 --num_resources 3 --sensor_range 1 --slope_angle 40 --arena_length 8 --arena_width 4 --cache_start 1 --slope_start 3 --source_start 7 --target_fitness 1.0

if __name__ == "__main__":
    main(sys.argv[1:])#Uncomment for proper runs
    #time_fitness_function()
    #time_rwg()