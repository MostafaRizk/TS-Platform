import os
import numpy as np
import json
import pandas as pd

from fitness import FitnessCalculator
from agents.nn_agent_lean import NNAgent
#from scripts.video_generator import get_video_from_model
from scipy.stats import multivariate_normal
from glob import glob


'''
evolved_genomes_parameters_directory = "/Users/mostafa/Documents/Code/PhD/TS-Platform/results/2020_11_24_magic_plot_combined_new_seed/experiments"
evolved_genomes_results_directory = "/Users/mostafa/Documents/Code/PhD/TS-Platform/results/2020_11_24_magic_plot_combined_new_seed/results"

other_param_dir = "/Users/mostafa/Documents/Code/PhD/TS-Platform/results/2020_11_10_magic_plot_shortened_episodes_evolution/experiments"
other_result_dir = "/Users/mostafa/Documents/Code/PhD/TS-Platform/results/2020_11_10_magic_plot_shortened_episodes_evolution/results"

parameter_filename = os.path.join(evolved_genomes_parameters_directory, "cma_heterogeneous_individual_nn_slope_2942995347_2_4_1_4_8_4_1_3_7_1_3.0_0.2_2_1000_500_5_rnn_False_1_4_tanh_100_0.2_1000_0.001_200.0.json")
#parameter_filename = os.path.join(evolved_genomes_parameters_directory, "cma_heterogeneous_individual_nn_slope_1904615677_2_4_1_4_8_4_1_3_7_1_3.0_0.2_2_1000_500_5_rnn_False_1_4_tanh_100_0.2_1000_0.001_200.0.json")
#parameter_filename = os.path.join(other_param_dir, "cma_heterogeneous_individual_nn_slope_491264_2_4_1_4_8_4_1_3_7_1_3.0_0.2_2_1000_500_5_rnn_False_1_4_tanh_100_0.2_1000_0.001_200.0.json")
fitness_calculator = FitnessCalculator(parameter_filename)

model_name = "cma_heterogeneous_individual_nn_slope_2942995347_2_4_1_4_8_4_1_3_7_1_3.0_0.2_2_1000_500_5_rnn_False_1_4_tanh_100_0.2_1000_0.001_200.0_54977.4_final.npy"
#model_name = "cma_heterogeneous_individual_nn_slope_1904615677_2_4_1_4_8_4_1_3_7_1_3.0_0.2_2_1000_500_5_rnn_False_1_4_tanh_100_0.2_1000_0.001_200.0_60390.0_final.npy"
model_file = os.path.join(evolved_genomes_results_directory, model_name)
#model_name = "cma_heterogeneous_individual_nn_slope_491264_2_4_1_4_8_4_1_3_7_1_3.0_0.2_2_1000_500_5_rnn_False_1_4_tanh_100_0.2_1000_0.001_200.0_60938.0_final.npy"
#model_file = os.path.join(other_result_dir, model_name)
'''

'''
full_genome = np.load(model_file)
mid = int(len(full_genome) / 2)

genome_part_1 = full_genome[0:mid]
genome_part_2 = full_genome[mid:]
agent_1 = NNAgent(fitness_calculator.get_observation_size(), fitness_calculator.get_action_size(),
                  parameter_filename, genome_part_1)
agent_2 = NNAgent(fitness_calculator.get_observation_size(), fitness_calculator.get_action_size(),
                  parameter_filename, genome_part_2)
results = fitness_calculator.calculate_fitness(agent_list=[agent_1, agent_2], render=True, time_delay=0.1,
                                               measure_specialisation=True, logging=False, logfilename=None,
                                               render_mode="human")
specialisation_measures = np.mean(np.array(results["specialisation_list"]), axis=0)'''

'''genome_1 = np.load(model_file)
genome_2 = np.load(model_file)

agent_1 = NNAgent(fitness_calculator.get_observation_size(), fitness_calculator.get_action_size(), parameter_filename, genome_1)
agent_2 = NNAgent(fitness_calculator.get_observation_size(), fitness_calculator.get_action_size(), parameter_filename, genome_2)
results = fitness_calculator.calculate_fitness(agent_list=[agent_1, agent_2], render=False, time_delay=0,
                                               measure_specialisation=True, logging=False, logfilename=None,
                                               render_mode="human")
specialisation_measures = np.mean(np.array(results["specialisation_list"]), axis=0)'''

# Sample 2 individuals
'''
mean_array = np.load(model_file)
std = None

log_file = os.path.join(evolved_genomes_results_directory, "_".join(model_name.split("_")[:-1]) + ".log")
g = open(log_file, "r")
log_data = g.read().strip().split("\n")
g.close()

# If last line starts with Best, get line before
flag = True
i = -1
while flag:
    if log_data[i].startswith("Best"):
        std = float(log_data[i-1].strip().split()[4]) / 10
        flag = False
    else:
        i -= 1

seed = 2942995347

random_variable = multivariate_normal(mean=mean_array, cov=np.identity(len(mean_array)) * std)
team = random_variable.rvs(2, seed)

agent_1 = NNAgent(fitness_calculator.get_observation_size(), fitness_calculator.get_action_size(),
                  parameter_filename, team[0])
agent_2 = NNAgent(fitness_calculator.get_observation_size(), fitness_calculator.get_action_size(),
                  parameter_filename, team[1])
results = fitness_calculator.calculate_fitness(agent_list=[agent_1, agent_2], render=False, time_delay=0,
                                               measure_specialisation=True, logging=False, logfilename=None,
                                               render_mode="human")
specialisation_measures = np.mean(np.array(results["specialisation_list"]), axis=0)
'''

# Get one team
'''
full_genome = np.load(model_file)
mid = int(len(full_genome) / 2)
genome_part_1 = full_genome[0:mid]
genome_part_2 = full_genome[mid:]
agent_1 = NNAgent(fitness_calculator.get_observation_size(), fitness_calculator.get_action_size(),
                  parameter_filename, genome_part_1)
agent_2 = NNAgent(fitness_calculator.get_observation_size(), fitness_calculator.get_action_size(),
                  parameter_filename, genome_part_2)
results = fitness_calculator.calculate_fitness(agent_list=[agent_1, agent_2], render=True, time_delay=0.1,
                                               measure_specialisation=True, logging=False, logfilename=None,
                                               render_mode="human")
specialisation_measures = np.mean(np.array(results["specialisation_list"]), axis=0)
'''

'''
parameter_dictionary = json.loads(open(parameter_filename).read())
parameter_dictionary["environment"]["slope"]["num_episodes"] = 1
f = open(f"temp.json", "w")
dictionary_string = json.dumps(parameter_dictionary, indent=4)
f.write(dictionary_string)
f.close()

get_video_from_model(model_name=model_file,parameter_filename="temp.json",save_directory="/Users/mostafa",
                     filename="bad_specialist.gif")
'''

'''
results_file = os.path.join(evolved_genomes_results_directory,f"results_final.csv")

# Load all evolved genomes into pandas dataframe
evolved_data = pd.read_csv(results_file)


f = open("team_sampling_smaller_sigma.log", "w")

# Get evolved data
# For each genome
for index, row in evolved_data.iterrows():
    ''''''
    if row["reward_level"] == "team" and row["num_agents"] == 2:
            # Get genome from corresponding model file
            model_file = os.path.join(evolved_genomes_results_directory, row["model_name"])

            parameter_filename = os.path.join(evolved_genomes_parameters_directory, "_".join(row["model_name"].split("_")[:-2]) + ".json")
            fitness_calculator = FitnessCalculator(parameter_filename)

            full_genome = np.load(model_file)
            mid = int(len(full_genome) / 2)
            genome_part_1 = full_genome[0:mid]
            genome_part_2 = full_genome[mid:]
            agent_1 = NNAgent(fitness_calculator.get_observation_size(), fitness_calculator.get_action_size(),
                              parameter_filename, genome_part_1)
            agent_2 = NNAgent(fitness_calculator.get_observation_size(), fitness_calculator.get_action_size(),
                              parameter_filename, genome_part_2)
            results = fitness_calculator.calculate_fitness(agent_list=[agent_1, agent_2], render=False, time_delay=0,
                                                           measure_specialisation=True, logging=False, logfilename=None,
                                                           render_mode="human")

            og_fitness = row["fitness"]
            fitness_matrix = results["fitness_matrix"]
            new_fitness = np.mean([fitness_matrix[0][episode] + fitness_matrix[1][episode] for episode in range(len(fitness_matrix[0]))])
            specialisation_measures = np.mean(np.array(results["specialisation_list"]), axis=0)
            model_name = row["model_name"]
            parameters = "_".join(row["model_name"].split("_")[:-2]) + ".json"

    ''''''
    # Duplicate genome if it has an individual reward
    if row["reward_level"] == "individual" and row["num_agents"] == 2:
        # Get genome from corresponding model file
        model_file = os.path.join(evolved_genomes_results_directory, row["model_name"])

        parameter_filename = os.path.join(evolved_genomes_parameters_directory, "_".join(row["model_name"].split("_")[:-2]) + ".json")
        fitness_calculator = FitnessCalculator(parameter_filename)

        genome_1 = np.load(model_file)
        genome_2 = np.load(model_file)

        agent_1 = NNAgent(fitness_calculator.get_observation_size(), fitness_calculator.get_action_size(), parameter_filename, genome_1)
        agent_2 = NNAgent(fitness_calculator.get_observation_size(), fitness_calculator.get_action_size(), parameter_filename, genome_2)
        results = fitness_calculator.calculate_fitness(agent_list=[agent_1, agent_2], render=False, time_delay=0,
                                                       measure_specialisation=True, logging=False, logfilename=None,
                                                       render_mode="human")
        og_fitness = 2*row["fitness"]
        fitness_matrix = results["fitness_matrix"]
        new_fitness = np.mean([fitness_matrix[0][episode] + fitness_matrix[1][episode] for episode in range(len(fitness_matrix[0]))])
        specialisation_measures = np.mean(np.array(results["specialisation_list"]), axis=0)

        parameters = "_".join(row["model_name"].split("_")[:-2]) + ".json"
        model_name = row["model_name"]

    # Sampling individuals
    ''''''
    if row["reward_level"] == "individual" and row["num_agents"] == 2:
        # Get genome from corresponding model file
        model_file = os.path.join(evolved_genomes_results_directory, row["model_name"])

        parameter_filename = os.path.join(evolved_genomes_parameters_directory,
                                          "_".join(row["model_name"].split("_")[:-2]) + ".json")
        fitness_calculator = FitnessCalculator(parameter_filename)

        mean_array = np.load(model_file)
        std = None

        log_file = os.path.join(evolved_genomes_results_directory, "_".join(row["model_name"].split("_")[:-1]) + ".log")
        g = open(log_file, "r")
        log_data = g.read().strip().split("\n")
        g.close()

        # If last line starts with Best, get line before
        flag = True
        i = -1
        while flag:
            if log_data[i].startswith("Best"):
                std = float(log_data[i-1].strip().split()[4]) / 10
                flag = False
            else:
                i -= 1

        seed = row["seed"]

        random_variable = multivariate_normal(mean=mean_array, cov=np.identity(len(mean_array)) * std)
        team = random_variable.rvs(2, seed)

        agent_1 = NNAgent(fitness_calculator.get_observation_size(), fitness_calculator.get_action_size(),
                          parameter_filename, team[0])
        agent_2 = NNAgent(fitness_calculator.get_observation_size(), fitness_calculator.get_action_size(),
                          parameter_filename, team[1])
        results = fitness_calculator.calculate_fitness(agent_list=[agent_1, agent_2], render=False, time_delay=0,
                                                       measure_specialisation=True, logging=False, logfilename=None,
                                                       render_mode="human")
        og_fitness = 2 * row["fitness"]
        fitness_matrix = results["fitness_matrix"]
        new_fitness = np.mean(
            [fitness_matrix[0][episode] + fitness_matrix[1][episode] for episode in range(len(fitness_matrix[0]))])
        specialisation_measures = np.mean(np.array(results["specialisation_list"]), axis=0)

        parameters = "_".join(row["model_name"].split("_")[:-2]) + ".json"
        model_name = row["model_name"]

    ''''''
    # Sampling teams
    if row["reward_level"] == "team" and row["num_agents"] == 2:
        # Get genome from corresponding model file
        model_file = os.path.join(evolved_genomes_results_directory, row["model_name"])

        parameter_filename = os.path.join(evolved_genomes_parameters_directory,
                                          "_".join(row["model_name"].split("_")[:-2]) + ".json")
        fitness_calculator = FitnessCalculator(parameter_filename)

        mean_array = np.load(model_file)
        std = None

        log_file = os.path.join(evolved_genomes_results_directory, "_".join(row["model_name"].split("_")[:-1]) + ".log")
        g = open(log_file, "r")
        log_data = g.read().strip().split("\n")
        g.close()

        # If last line starts with Best, get line before
        flag = True
        i = -1
        while flag:
            if log_data[i].startswith("Best"):
                std = float(log_data[i - 1].strip().split()[4])/10
                flag = False
            else:
                i -= 1

        # 1000 100000 -1.998880000000000e+04 2.4e+00 4.46e-02  4e-02  5e-02 526:49.5
        seed = row["seed"]

        random_variable = multivariate_normal(mean=mean_array, cov=np.identity(len(mean_array)) * std)
        full_genome = random_variable.rvs(1, seed)

        mid = int(len(full_genome) / 2)
        genome_part_1 = full_genome[0:mid]
        genome_part_2 = full_genome[mid:]
        agent_1 = NNAgent(fitness_calculator.get_observation_size(), fitness_calculator.get_action_size(),
                          parameter_filename, genome_part_1)
        agent_2 = NNAgent(fitness_calculator.get_observation_size(), fitness_calculator.get_action_size(),
                          parameter_filename, genome_part_2)
        results = fitness_calculator.calculate_fitness(agent_list=[agent_1, agent_2], render=False, time_delay=0,
                                                       measure_specialisation=True, logging=False, logfilename=None,
                                                       render_mode="human")

        og_fitness = row["fitness"]
        fitness_matrix = results["fitness_matrix"]
        new_fitness = np.mean(
            [fitness_matrix[0][episode] + fitness_matrix[1][episode] for episode in range(len(fitness_matrix[0]))])
        specialisation_measures = np.mean(np.array(results["specialisation_list"]), axis=0)
        model_name = row["model_name"]
        parameters = "_".join(row["model_name"].split("_")[:-2]) + ".json"

        f.write(f"Original fitness: {og_fitness}\n")
        f.write(f"Recalculated fitness: {new_fitness}\n")
        f.write(f"Recalculated specialisation: {specialisation_measures}\n")
        f.write("\n")
        f.write(f"Recalculated fitness matrix: {fitness_matrix}\n")
        f.write(f"Model File: {model_name}\n")
        f.write(f"Parameter file: {parameters}\n")
        f.write("-----\n")

f.close()
'''

def sample_individual(path_to_genome):
    model_name = path_to_genome.split("/")[-1]
    main_folder = "/".join(str(param) for param in path_to_genome.split("/")[:-2])
    experiments_folder = os.path.join(main_folder, "experiments")
    results_folder = os.path.join(main_folder, "results")

    parameter_filename = os.path.join(experiments_folder, "cma_with_seeding_" + "_".join(model_name.split("/")[-1].split("_")[1:-2]) + ".json")
    fitness_calculator = FitnessCalculator(parameter_filename)

    mean_array = np.load(path_to_genome)
    variance = None

    name_substring = "_".join(str(param) for param in model_name.split("/")[-1].split("_")[1:-2])
    regex_string = f'{results_folder}/cma_with_seeding_*{name_substring}*.log'
    log_files = glob(regex_string)

    if len(log_files) > 1 or len(log_files) < 1:
        raise RuntimeError("Inappropriate number of log files")

    log_file = log_files[0]
    g = open(log_file, "r")
    log_data = g.read().strip().split("\n")
    g.close()

    # If last line starts with Best, get line before
    flag = True
    i = -1
    while flag:
        if log_data[i].startswith("Best"):
            variance = float(log_data[i - 1].strip().split()[4]) / 10
            flag = False
        else:
            i -= 1

    # 1000 100000 -1.998880000000000e+04 2.4e+00 4.46e-02  4e-02  5e-02 526:49.5
    seed = int(model_name.split("/")[-1].split("_")[5])

    random_variable = multivariate_normal(mean=mean_array, cov=np.identity(len(mean_array)) * variance)
    team = random_variable.rvs(2, seed)
    agent_1 = NNAgent(fitness_calculator.get_observation_size(), fitness_calculator.get_action_size(), parameter_filename, team[0])
    agent_2 = NNAgent(fitness_calculator.get_observation_size(), fitness_calculator.get_action_size(), parameter_filename, team[1])
    results = fitness_calculator.calculate_fitness(agent_list=[agent_1, agent_2], render=True, time_delay=0.1,
                                                   measure_specialisation=True, logging=False, logfilename=None,
                                                   render_mode="human")

path = "/Users/mostafa/Documents/Code/PhD/TS-Platform/results/2021_02_17_cma_for_diff_slopes_combined/results/cma_heterogeneous_individual_nn_slope_550290314_2_4_1_0_8_4_1_3_7_1_3.0_0.2_2_1000_500_20_rnn_False_1_4_tanh_100_0.2_1000_0.001_0.0_13926.0_final.npy"
sample_individual(path)