import sys
import getopt
import os
import numpy as np

from gym_TS.agents.TinyAgent import TinyAgent
from gym_TS.fitness_calculator import FitnessCalculator
from gym_TS.learning_algorithms import cma_es, rwg


def main(argv):
    try:
        opts, args = getopt.getopt(argv, "",
                                   ["algorithm=", "team_type=", "selection_level=", "generations=", "simulation_length=", "trials=",
                                    "seed=", "num_robots=", "num_resources=", "sensor_range=", "slope_angle=",
                                    "arena_length=", "arena_width=", "cache_start=", "slope_start=", "source_start=",
                                    "sigma=", "test_model=", "target_fitness=", "num_teams=", "batch_test=",
                                    "hardcoded_test=", "upward_cost_factor=", "downward_cost_factor=", "carry_factor=",
                                    "resource_reward_factor="])

    except getopt.GetoptError:
        print("There was an error with the given arguments")
        sys.exit(2)

    bootstrap = False

    training_algorithm = None
    team_type = None
    selection_level = None
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
    upward_cost_factor = None
    downward_cost_factor = None
    carry_factor = None
    resource_reward_factor = None
    sigma = None
    test_model = None
    target_fitness = 1.0
    num_teams = None
    batch_path = None
    hardcoded_test = None

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
        if opt == "--selection_level":
            if arg == "team" or arg == "individual":
                selection_level = arg
            else:
                print("Selection level is misspelled or unsupported. Use \"team\" or \"individual\"")
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
        if opt == "--upward_cost_factor":
            upward_cost_factor = float(arg)
        if opt == "--downward_cost_factor":
            downward_cost_factor = float(arg)
        if opt == "--carry_factor":
            carry_factor = float(arg)
        if opt == "--resource_reward_factor":
            resource_reward_factor = float(arg)
        if opt == "--sigma":
            sigma = float(arg)
        if opt == "--test_model":
            test_model = arg
        if opt == "--target_fitness":
            target_fitness = float(arg)
        if opt == "--num_teams":
            num_teams = int(arg)
        if opt == "--batch_test":
            batch_path = arg
        if opt == "--hardcoded_test":
            hardcoded_test = arg

    if bootstrap:
        fitness_calculator = FitnessCalculator(random_seed=random_seed, simulation_length=simulation_length,
                                               num_trials=num_trials, num_robots=num_robots,
                                               num_resources=num_resources,
                                               sensor_range=sensor_range, slope_angle=slope_angle,
                                               arena_length=arena_length, arena_width=arena_width,
                                               cache_start=cache_start,
                                               slope_start=slope_start, source_start=source_start,
                                               upward_cost_factor=upward_cost_factor,
                                               downward_cost_factor=downward_cost_factor, carry_factor=carry_factor,
                                               resource_reward_factor=resource_reward_factor)

        best_genome, best_fitness = training_algorithm(seed_value=random_seed, calculator=fitness_calculator,
                                                       num_teams=num_generations, team_type=team_type,
                                                       selection_level=selection_level, target_fitness=target_fitness)

        model_name = f"bootstrap_{team_type}_{selection_level}_{simulation_length}_{num_trials}_{random_seed}_{num_robots}_{num_resources}_{sensor_range}_{slope_angle}_{arena_length}_{arena_width}_{cache_start}_{slope_start}_{source_start}_{upward_cost_factor}_{downward_cost_factor}_{carry_factor}_{resource_reward_factor}"

        np.save(model_name, best_genome)

    else:
        # If this is a training run
        if test_model is None and batch_path is None and hardcoded_test is None:

            # Prepare fitness function
            fitness_calculator = FitnessCalculator(random_seed=random_seed, simulation_length=simulation_length,
                                                   num_trials=num_trials, num_robots=num_robots,
                                                   num_resources=num_resources,
                                                   sensor_range=sensor_range, slope_angle=slope_angle,
                                                   arena_length=arena_length, arena_width=arena_width,
                                                   cache_start=cache_start,
                                                   slope_start=slope_start, source_start=source_start,
                                                   upward_cost_factor=upward_cost_factor,
                                                   downward_cost_factor=downward_cost_factor, carry_factor=carry_factor,
                                                   resource_reward_factor=resource_reward_factor)

            model_name = f"CMA_{team_type}_{selection_level}_{simulation_length}_{num_generations}_{num_trials}_{random_seed}_{num_robots}_{num_resources}_{sensor_range}_{slope_angle}_{arena_length}_{arena_width}_{cache_start}_{slope_start}_{source_start}_{upward_cost_factor}_{downward_cost_factor}_{carry_factor}_{resource_reward_factor}_{sigma}_{num_teams}"

            # Create results file if it doesn't exist
            results_file_name = "results.csv"

            if os.path.exists(results_file_name):
                pass
            else:
                results_file = open(results_file_name, 'w')
                results_file.write("Algorithm Name, Team Type, Selection Level, Simulation Length, Num Generations, Num Trials, Random Seed, Num Robots, Num Resources, Sensor Range, Slope Angle, Arena Length, Arena Width, Cache Start, Slope Start, Source Start, Upward cost factor, Downward cost factor, Carry Factor, Resource reward factor, Sigma, Population, Log File, Seed Fitness, Evolved Fitness\n")
                results_file.close()

            print(f"Evaluating {model_name}")

            # Get best genome using CMA
            best_genome = training_algorithm(fitness_calculator=fitness_calculator, seed_value=random_seed, sigma=sigma,
                                             model_name=model_name, results_file_name=results_file_name,
                                             team_type=team_type, selection_level=selection_level, num_generations=num_generations,
                                             num_teams=num_teams)

            if team_type == "heterogeneous" and selection_level == "team":
                best_individual_1 = TinyAgent(fitness_calculator.get_observation_size(),
                                              fitness_calculator.get_action_size(),
                                              seed=random_seed)
                best_individual_2 = TinyAgent(fitness_calculator.get_observation_size(),
                                              fitness_calculator.get_action_size(),
                                              seed=random_seed)

                # Split genome
                mid = int(len(best_genome) / 2)
                best_individual_1.load_weights(best_genome[0:mid])
                best_individual_2.load_weights(best_genome[mid:])

                best_individual_1.save_model(model_name + "_controller1_")
                best_individual_2.save_model(model_name + "_controller2_")

            else:
                best_individual = TinyAgent(fitness_calculator.get_observation_size(),
                                            fitness_calculator.get_action_size(),
                                            seed=random_seed)
                best_individual.load_weights(best_genome)
                best_individual.save_model(model_name)

        # If this is a testing run
        elif batch_path is None and hardcoded_test is None:
            model_name = test_model.split("_")
            # CMA_homogeneous_team_500_5000_5_30_2_3_1_40_8_4_1_3_7_3.0_0.2_2.0_1000.0_0.2_40.npy
            # f"CMA_{team_type}_{selection_level}_{simulation_length}_{num_generations}_{num_trials}_{random_seed}_{num_robots}_{num_resources}_{sensor_range}_{slope_angle}_{arena_length}_{arena_width}_{cache_start}_{slope_start}_{source_start}_{upward_cost_factor}_{downward_cost_factor}_{carry_factor}_{resource_reward_factor}_{sigma}_{num_teams}"

            # Prepare fitness function
            fitness_calculator = FitnessCalculator(random_seed=int(model_name[7]), simulation_length=int(model_name[4]),
                                                   num_trials=5, num_robots=int(model_name[8]),
                                                   num_resources=int(model_name[9]),
                                                   sensor_range=int(model_name[10]), slope_angle=int(model_name[11]),
                                                   arena_length=int(model_name[12]), arena_width=int(model_name[13]),
                                                   cache_start=int(model_name[14]),
                                                   slope_start=int(model_name[15]), source_start=int(model_name[16]),
                                                   upward_cost_factor=float(model_name[17]),
                                                   downward_cost_factor=float(model_name[18]), carry_factor=float(model_name[19]),
                                                   resource_reward_factor=float(model_name[20]),
                                                   using_gym=True)

            team_type = model_name[2]
            selection_level = model_name[3]

            individual_1 = None
            individual_2 = None

            if team_type == "heterogeneous" and selection_level == "team":
                if model_name[23] != "controller1":
                    raise RuntimeError("Use controller 1's path")

                individual_1 = np.load(test_model)
                test_model2 = test_model.replace("controller1", "controller2")
                individual_2 = np.load(test_model2)

            elif team_type == "homogeneous" and selection_level == "team" or \
                    team_type == "heterogeneous" and selection_level == "individual" or \
                    team_type == "homogeneous" and selection_level == "individual":
                individual_1 = np.load(test_model)
                individual_2 = np.load(test_model)

            else:
                raise RuntimeError("Invalid team type and/or selection level")


            #fitness, specialisation = fitness_calculator.calculate_ferrante_specialisation(full_genome, team_type=model_name[2],render=False)
            #print(f"Fitness is {fitness} and specialisation is {specialisation}")
            fitness_1, fitness_2 = fitness_calculator.calculate_fitness(team_type=model_name[2], selection_level=model_name[3], individual_1=individual_1, individual_2=individual_2, render=True)
            print(f"Fitness 1 is {fitness_1} and Fitness 2 is {fitness_2}")

        # If this is a batch testing run
        elif batch_path is not None:

            # Create results file if it doesn't exist
            results_file_name = batch_path+"/results_specialisation.csv"
            results_file = None

            if os.path.exists(results_file_name):
                pass
            else:
                results_file = open(results_file_name, 'w')
                results_file.write("Algorithm Name, Team Type, Selection Level, Simulation Length, Num Generations, Num Trials, Random Seed, Num Robots, Num Resources, Sensor Range, Slope Angle, Arena Length, Arena Width, Cache Start, Slope Start, Source Start, Upward cost factor, Downward cost factor, Carry Factor, Resource reward factor, Sigma, Population, Fitness 1, Fitness 2, Team Fitness, Specialisation\n")

            for filename in os.listdir(batch_path):

                if filename.endswith(".npy") and filename.startswith("CMA"):
                    model_name = filename.split("_")
                    if len(model_name) == 23 and model_name[22] != "controller1" and model_name[22] != "controller2":
                        continue
                    # CMA_homogeneous_team_500_5000_5_30_2_3_1_40_8_4_1_3_7_3.0_0.2_2.0_1000.0_0.2_40.npy
                    # f"CMA_{team_type}_{selection_level}_{simulation_length}_{num_generations}_{num_trials}_{random_seed}_{num_robots}_{num_resources}_{sensor_range}_{slope_angle}_{arena_length}_{arena_width}_{cache_start}_{slope_start}_{source_start}_{upward_cost_factor}_{downward_cost_factor}_{carry_factor}_{resource_reward_factor}_{sigma}_{num_teams}"

                    # Prepare fitness function
                    # Prepare fitness function
                    fitness_calculator = FitnessCalculator(random_seed=int(model_name[6]),
                                                           simulation_length=int(model_name[3]),
                                                           num_trials=5, num_robots=int(model_name[7]),
                                                           num_resources=int(model_name[8]),
                                                           sensor_range=int(model_name[9]),
                                                           slope_angle=int(model_name[10]),
                                                           arena_length=int(model_name[11]),
                                                           arena_width=int(model_name[12]),
                                                           cache_start=int(model_name[13]),
                                                           slope_start=int(model_name[14]),
                                                           source_start=int(model_name[15]),
                                                           upward_cost_factor=float(model_name[16]),
                                                           downward_cost_factor=float(model_name[17]),
                                                           carry_factor=float(model_name[18]),
                                                           resource_reward_factor=float(model_name[19]),
                                                           using_gym=True)

                    team_type = model_name[1]
                    selection_level = model_name[2]

                    test_model = batch_path + "/" + filename

                    individual_1 = None
                    individual_2 = None

                    if team_type == "heterogeneous" and selection_level == "team":
                        if model_name[22] != "controller1":
                            raise RuntimeError("Use controller 1's path")

                        individual_1 = np.load(test_model)
                        test_model2 = test_model.replace("controller1", "controller2")
                        individual_2 = np.load(test_model2)

                    elif team_type == "homogeneous" and selection_level == "team" or \
                            team_type == "heterogeneous" and selection_level == "individual" or \
                            team_type == "homogeneous" and selection_level == "individual":
                        individual_1 = np.load(test_model)
                        individual_2 = np.load(test_model)

                    else:
                        raise RuntimeError("Invalid team type and/or selection level")

                    fitness_1, fitness_2, specialisation = fitness_calculator.calculate_ferrante_specialisation(team_type=model_name[1], selection_level=model_name[2], individual_1=individual_1, individual_2=individual_2, render=False)

                    if specialisation >= 0.0:
                        print(f"Fitnesses are {fitness_1} and {fitness_2}. Specialisation is {specialisation}")
                        print(filename)

                    results = filename.strip(".npy").replace("_", ",")
                    results += f", {fitness_1}, {fitness_2}, {fitness_1+fitness_2}, {specialisation}\n"
                    results_file = open(results_file_name, 'a')
                    results_file.write(results)

            results_file.close()

        elif hardcoded_test is not None:
            fitness_calculator = FitnessCalculator(random_seed=random_seed, simulation_length=simulation_length,
                                                   num_trials=num_trials, num_robots=num_robots,
                                                   num_resources=num_resources,
                                                   sensor_range=sensor_range, slope_angle=slope_angle,
                                                   arena_length=arena_length, arena_width=arena_width,
                                                   cache_start=cache_start,
                                                   slope_start=slope_start, source_start=source_start,
                                                   upward_cost_factor=upward_cost_factor,
                                                   downward_cost_factor=downward_cost_factor, carry_factor=carry_factor,
                                                   resource_reward_factor=resource_reward_factor, using_gym=False)

            fitness = 0
            specialisation = 0

            # Create results file if it doesn't exist
            results_file_name = "hardcoded_results.csv"
            results_file = None

            if os.path.exists(results_file_name):
                results_file = open(results_file_name, 'a')
            else:
                results_file = open(results_file_name, 'w')
                results_file.write("Strategy,Individual 1 Score,Individual 2 Score,Seed\n")

            fitness_1, fitness_2, specialisation = fitness_calculator.calculate_hardcoded_fitness(type=hardcoded_test, render=True)

            results_file.write(f"{hardcoded_test},{fitness_1},{fitness_2},{random_seed}\n")

            results_file.close()

            print(f"Seed {random_seed}: {hardcoded_test}")

            #print(f"Fitness 1 is {fitness_1}, Fitness 2 is {fitness_2}, Team Fitness is {fitness_1+fitness_2} and specialisation is {specialisation}")

# To run, use:
# python3 main.py --algorithm bootstrap --team_type homogeneous --generations 500 --simulation_length 500 --trials 5  --seed 1 --num_robots 2 --num_resources 3 --sensor_range 1 --slope_angle 40 --arena_length 8 --arena_width 4 --cache_start 1 --slope_start 3 --source_start 7 --upward_cost_factor 3 --downward_cost_factor 0.2 --carry_factor 10 --resource_reward_factor 1000 --target_fitness 100
# python3 main.py --algorithm cma --team_type homogeneous --generations 5000 --simulation_length 500 --trials 5  --seed 1 --num_robots 2 --num_resources 3 --sensor_range 1 --slope_angle 40 --arena_length 8 --arena_width 4 --cache_start 1 --slope_start 3 --source_start 7 --upward_cost_factor 3 --downward_cost_factor 0.2 --carry_factor 10 --resource_reward_factor 1000 --sigma 0.2 --population 40
# python3 main.py --test_model /home/mriz9/Documents/Results/AAMAS/8_SpeedCost/models/Tiny/CMA_heterogeneous_500_300_5_10_2_3_1_40_16_4_1_3_15_0.05_40_controller1_.npy
# python3 main.py --batch_test /home/mriz9/Documents/Results/AAMAS/8_SpeedCost/models/Tiny
# python3 main.py --hardcoded_test generalist --simulation_length 500 --trials 5 --seed 1 --num_robots 2 --num_resources 3 --sensor_range 1 --slope_angle 40 --arena_length 8 --arena_width 4 --cache_start 1 --slope_start 3 --source_start 7

if __name__ == "__main__":
    main(sys.argv[1:])  # Uncomment for proper runs
    # time_fitness_function()
    # time_rwg()
