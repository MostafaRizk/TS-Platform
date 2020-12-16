import numpy as np
import json
import os

from learning.learner_parent import Learner
from learning.rwg import RWGLearner
from glob import glob


def get_best_of_k_samples(results_file, parameter_filename, k, N_episodes):
    """
    From a results file with n samples. Find the best among the first k samples and write it to a file

    @param results_file: CSV file containing n models
    @param parameter_filename: Parameter file used to generate results
    @param k: An integer
    @param N_episodes: Number of episodes used to evaluate the sample
    """

    # Load json and csv files
    parameter_dictionary = json.loads(open(parameter_filename).read())
    f = open(results_file, "r")
    data = f.read().strip().split("\n")

    # Get best genome in first k samples
    best_genome = np.array([float(element) for element in data[0].split(",")[0:-N_episodes]])
    best_score = np.mean([float(episode_score) for episode_score in data[0].split(",")[-N_episodes:]])

    for row in data[0:k]:
        genome = np.array([float(element) for element in row.split(",")[0:-N_episodes]])
        episode_scores = [float(score) for score in row.split(",")[-N_episodes:]]
        mean_score = np.mean(episode_scores)

        if mean_score > best_score:
            best_score = mean_score
            best_genome = genome

    # Generate model name
    if parameter_dictionary['general']['reward_level'] == "team":
        parameter_dictionary['algorithm']['agent_population_size'] = k * 2
    else:
        parameter_dictionary['algorithm']['agent_population_size'] = k

    parameters_in_name = Learner.get_core_params_in_model_name(parameter_dictionary)
    parameters_in_name += RWGLearner.get_additional_params_in_model_name(parameter_dictionary)
    parameters_in_name += [best_score]
    model_name = "_".join([str(param) for param in parameters_in_name])

    np.save(model_name, best_genome)

def copy_all_models(source_directory, destination_directory):
    npy_files = glob(f'{source_directory}/cma*npy')
    #os.copy

def fix_results(results_folder, start_generation, num_generations, step_size):
    correct_header = "algorithm_selected,team_type,reward_level,agent_type,environment,seed,num_agents,num_resources,sensor_range,sliding_speed,arena_length,arena_width,cache_start,slope_start,source_start,base_cost,upward_cost_factor,downward_cost_factor,carry_factor,resource_reward,episode_length,num_episodes,architecture,bias,hidden_layers,hidden_units_per_layer,activation_function,agent_population_size,sigma,generations,tolx,tolfunhist,seed_fitness,fitness,model_name"

    """for i in range(start_generation, num_generations+1, step_size):
        results_file = f"{results_folder}/results_{i}.csv"

        f = open(results_file)
        file_text = f.read().split('\n')
        f.close()
        file_text[0] = correct_header
        file_text = '\n'.join(file_text)
        f = open(results_file, 'w')
        f.write(file_text)
        f.close()"""

    results_file = f"{results_folder}/results_final.csv"

    f = open(results_file)
    file_text = f.read().split('\n')
    f.close()
    file_text[0] = correct_header
    file_text = '\n'.join(file_text)
    f = open(results_file, 'w')
    f.write(file_text)
    f.close()

def get_seed_file(results_folder, parameter_list):
    """
    Get seed file that matches a particular parameter list

    @param results_folder:
    @param parameter_list:
    @return:
    """
    # Get pre and post seed params for model
    og_dir = os.getcwd()
    print(og_dir)
    seed_folder = f'{results_folder}/results'
    os.chdir(seed_folder)
    parameter_list[0] = "rwg"
    parameter_list = [parameter_list[0]] + parameter_list[3:27]

    # If individual reward, allow seeds that used any number of agents
    if parameter_list[2] == "individual":
        parameter_list[6] = '*'

    # Get list of all seed files with matching parameters
    seedfile_prefix = "_".join([str(param) for param in parameter_list])
    seedfile_extension = ".npy"
    possible_seedfiles = glob(f'{seedfile_prefix}*{seedfile_extension}')

    os.chdir(og_dir)

    # Return seed_file name but throw error if there's more than one
    if len(possible_seedfiles) == 0:
        raise RuntimeError('No valid seed files')
    elif len(possible_seedfiles) > 1:
        raise RuntimeError('Too many valid seed files')
    else:
        return possible_seedfiles[0]


def create_results_from_models(results_folder, start_generation, step_size, num_generations):
    """
    Assemble results csv files from numpy files

    Currently only works for results_final.csv
    @param results_folder:
    @return:
    """
    new_path = f'{results_folder}/results'
    os.chdir(new_path)
    generation_list = [i for i in range(start_generation, num_generations+step_size, step_size)]
    generation_list += ["final"]

    for generation in generation_list:
        # Get list of all final models
        model_files = glob(f'cma*_{generation}.npy')

        # Create final results file
        results_file = f'results_{generation}.csv'
        f = open(results_file, 'w')

        # Write header
        header = "algorithm_selected,team_type,reward_level,agent_type,environment,seed,num_agents,num_resources,sensor_range,sliding_speed,arena_length,arena_width,cache_start,slope_start,source_start,base_cost,upward_cost_factor,downward_cost_factor,carry_factor,resource_reward,episode_length,num_episodes,architecture,bias,hidden_layers,hidden_units_per_layer,activation_function,agent_population_size,sigma,generations,tolx,tolfunhist,seed_fitness,fitness,model_name"
        f.write(header)
        f.write("\n")

        # For every model, extract parameters and convert to a comma separated list
        for model_name in model_files:
            parameter_list = model_name.split("_")[0:-2]
            fitness = model_name.split("_")[-2]
            path_to_results = f'../'
            seed_file = get_seed_file(path_to_results, parameter_list)
            seed_fitness = seed_file.split("_")[-1].strip(".npy")

            # Log parameters and score to results file
            parameters_to_log = parameter_list + [seed_fitness] + [fitness] + [model_name]
            line_to_log = ",".join(parameters_to_log)
            f.write(line_to_log)
            f.write(("\n"))

        # Close results file
        f.close()

#get_best_of_k_samples(
#    "all_genomes_rwg_heterogeneous_team_nn_slope_1_2_4_1_4_8_4_1_3_7_1_3.0_0.2_2_1000_*.csv')_500_20_rnn_False_1_4_tanh_20000_normal_0_1_.csv",
#    "rnn_no-bias_1HL_4HU_tanh.json", k=1000, N_episodes=20)


# fix_results(results_folder="../../results/2020_11_08_2_agents_with varied_episodes/results_final.csv", start_generation=20, num_generations=1000, step_size=20)


create_results_from_models("../../results/2020_12_09_unique_seeds", start_generation=20, step_size=20, num_generations=1000)
