import sys
import getopt
import os

from agents.DQNAgent import DQNAgent
from agents.BasicQAgent import BasicQAgent

from agents.TinyAgent import TinyAgent
from fitness_calculator import FitnessCalculator

from learning_algorithms import cma_es


def main(argv):
    try:
        opts, args = getopt.getopt(argv, "",
                                   ["algorithm=", "simulation_length=", "trials=",
                                    "seed=", "num_robots=", "num_resources=", "sensor_range=", "slope_angle=",
                                    "arena_length=", "arena_width=", "cache_start=", "slope_start=", "source_start=", "sigma="])

    except getopt.GetoptError:
        print("There was an error")
        sys.exit(2)

    training_algorithm = None
    simulation_length = None
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

    # Read in arguments
    for opt, arg in opts:
        if opt == "--algorithm":
            if arg == "cma":
                training_algorithm = cma_es
            else:
                print("This algorithm is either misspelled or not supported yet")
                sys.exit(2)
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

    # Prepare fitness function
    fitness_calculator = FitnessCalculator(random_seed=random_seed, simulation_length=simulation_length,
                                           num_trials=num_trials, num_robots=num_robots, num_resources=num_resources,
                                           sensor_range=sensor_range, slope_angle=slope_angle,
                                           arena_length=arena_length, arena_width=arena_width, cache_start=cache_start,
                                           slope_start=slope_start, source_start=source_start)

    model_name = f"CMA_{simulation_length}_{num_trials}_{random_seed}_{num_robots}_{num_resources}_{sensor_range}_{slope_angle}_{arena_length}_{arena_width}_{cache_start}_{slope_start}_{source_start}_{sigma}"

    # Create results file if it doesn't exist
    results_file_name = "results.csv"

    if os.path.exists(results_file_name):
        pass
    else:
        results_file = open(results_file_name, 'w')
        results_file.write("Algorithm Name, Simulation Length, Num Trials, Random Seed, Num Robots, Num Resources, Sensor Range, Slope Angle, Arena Length, Arena Width, Cache Start, Slope Start, Source Start, Sigma, Log File, Fitness\n")
        results_file.close()

    print(f"Evaluating {model_name}")

    # Get best genome using CMA
    best_genome = training_algorithm(fitness_calculator=fitness_calculator, seed_value=random_seed, sigma=sigma, model_name=model_name, results_file_name=results_file_name)

    # Create individual using genome so that it can be saved
    best_individual = TinyAgent(fitness_calculator.get_observation_size(), fitness_calculator.get_action_size(),
                                seed=random_seed)
    best_individual.load_weights(best_genome)
    best_individual.save_model(model_name)


# To run, use:
# python3 gym_TS/main.py --algorithm cma --simulation_length 1000 --trials 1  --seed 1 --num_robots 2 --num_resources 3 --sensor_range 1 --slope_angle 20 --arena_length 8 --arena_width 4 --cache_start 1 --slope_start 3 --source_start 7 --sigma 0.01
if __name__ == "__main__":
    main(sys.argv[1:])
