import os
import json
import numpy as np
import argparse

from glob import glob
from learning.learner_parent import Learner
from evaluation.evaluate_model import evaluate_model


def get_seed_file(data_folder, parameter_dictionary):
    """
    Get seed file that matches a particular parameter dictionary
    """
    # Get pre and post seed params for model
    #parameter_list = ["centralised_rwg"] + parameter_list[2:-7]
    parameter_dictionary["general"]["learning_type"] = "centralised"
    parameter_dictionary["general"]["algorithm_selected"] = "rwg"

    # If individual reward, allow seeds that used any number of agents
    if parameter_dictionary["general"]["reward_level"] == "individual":
        parameter_dictionary["environment"]["slope"]["num_agents"] = 1

    # Get list of all seed files with matching parameters
    seedfile_prefix = "_".join(str(item) for item in Learner.get_core_params_in_model_name(parameter_dictionary))
    seedfile_extension = ".npy"
    possible_seedfiles = glob(f'{data_folder}/{seedfile_prefix}*{seedfile_extension}')

    # Return seed_file name but throw error if there's more than one
    if len(possible_seedfiles) == 0:
        raise RuntimeError('No valid seed files')
    elif len(possible_seedfiles) > 1:
        raise RuntimeError('Too many valid seed files')
    else:
        return possible_seedfiles[0]


def create_results_from_models(path_to_data_folder, start_generation, step_size, num_generations):
    """
    Assemble results csv files from numpy files
    """
    generation_list = [i for i in range(start_generation, num_generations+step_size, step_size)]
    generation_list += ["final"]

    for generation in generation_list:
        print(f"Doing results for generation {generation}")
        # Get list of all final models
        model_files = glob(f'{path_to_data_folder}/*cma*_{generation}.npy')  # TODO: Allow different algorithms

        # Create final results file
        results_file = os.path.join(path_to_data_folder, f'results_{generation}.csv')
        f = open(results_file, 'w')

        # Write header
        # TODO: Get the header using the learner classes
        header = "learning_type,algorithm_selected,team_type,reward_level,agent_type,environment,seed,num_agents,num_resources,sensor_range,sliding_speed,arena_length,arena_width,cache_start,slope_start,source_start,base_cost,upward_cost_factor,downward_cost_factor,carry_factor,resource_reward,episode_length,num_episodes, incremental_rewards,architecture,bias,hidden_layers,hidden_units_per_layer,activation_function,agent_population_size,sigma,generations,tolx,tolfunhist,tolflatfitness,tolfun,agent_index,seed_fitness,fitness,model_name"
        f.write(header)
        f.write("\n")

        # For every model, extract parameters and convert to a comma separated list
        for model_name in model_files:
            learning_type = model_name.split("/")[-1].split("_")[0]
            if learning_type == "centralised":
                parameter_list = model_name.split("/")[-1].split("_")[:-2]
                parameter_filename = "_".join(parameter_list) + ".json"
                agent_index = "None"
            elif learning_type == "decentralised":
                parameter_list = model_name.split("/")[-1].split("_")[:-3]
                parameter_filename = "_".join(parameter_list) + ".json"
                agent_index = model_name.split("/")[-1].split("_")[-3]
            else:
                raise RuntimeError("Model prefix must be Centralised or Decentralised")

            parameter_path = os.path.join(path_to_data_folder, parameter_filename)
            parameter_dictionary = json.loads(open(parameter_path).read())
            fitness = model_name.split("_")[-2]
            seed_file = get_seed_file(path_to_data_folder, parameter_dictionary)
            seed_fitness = seed_file.split("_")[-1].strip(".npy")

            # Log parameters and score to results file
            parameters_to_log = parameter_list + [agent_index] + [seed_fitness] + [fitness] + [model_name]
            line_to_log = ",".join(parameters_to_log)
            f.write(line_to_log)
            f.write("\n")

        # Close results file
        f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Store models in results file')
    parser.add_argument('--data_path', action="store")
    parser.add_argument('--start_generation', action="store")
    parser.add_argument('--step_size', action="store")
    parser.add_argument('--num_generations', action="store")

    data_path = parser.parse_args().data_path
    start_generation = parser.parse_args().start_generation
    step_size = parser.parse_args().step_size
    num_generations = parser.parse_args().num_generations

    create_results_from_models(data_path, start_generation, step_size, num_generations)