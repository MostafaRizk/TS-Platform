import os
import json
import numpy as np
import argparse

from glob import glob
from learning.learner_parent import Learner
from evaluation.evaluate_model import evaluate_model
from evaluation.create_results_from_models import get_seed_file


def create_reevaluated_results(path_to_data_folder, generation):
    # TODO: Modify to avoid repetition of other function
    # Get list of models
    model_files = glob(f'{path_to_data_folder}/*cma*_{generation}.npy')  # TODO: Allow different algorithms

    # Create final results file
    results_file = os.path.join(path_to_data_folder, f'results_reevaluated_{generation}.csv')
    f = open(results_file, 'w')

    # Write header
    # TODO: Get the header using the learner classes
    header = "learning_type,algorithm_selected,team_type,reward_level,agent_type,environment,seed,num_agents,num_resources,sensor_range,sliding_speed,arena_length,arena_width,cache_start,slope_start,source_start,base_cost,upward_cost_factor,downward_cost_factor,carry_factor,resource_reward,episode_length,num_episodes, incremental_rewards,architecture,bias,hidden_layers,hidden_units_per_layer,activation_function,agent_population_size,sigma,generations,tolx,tolfunhist,tolflatfitness,tolfun,agent_index,seed_fitness,fitness,seed_specialisation,specialisation,model_name"
    f.write(header)
    f.write("\n")

    # List of all evaluated decentralised teams
    # (Multiple members of the same team will be encountered in the loop,
    # this prevents the same team being evaluated multiple times)
    evaluated_teams = []

    # Get list of agent_scores for each
    for model_path in model_files:
        agent_scores, specialisation = evaluate_model(model_path)

        learning_type = model_path.split("/")[-1].split("_")[0]
        if learning_type == "centralised":
            parameter_list = model_path.split("/")[-1].split("_")[:-2]
            parameter_filename = "_".join(parameter_list) + ".json"
            agent_index = "None"
        elif learning_type == "decentralised":
            parameter_list = model_path.split("/")[-1].split("_")[:-3]

            team_prefix = "_".join(parameter_list)
            if team_prefix in evaluated_teams:
                continue

            parameter_filename = "_".join(parameter_list) + ".json"
            agent_index = model_path.split("/")[-1].split("_")[-3]
        else:
            raise RuntimeError("Model prefix must be Centralised or Decentralised")

        parameter_path = os.path.join(path_to_data_folder, parameter_filename)
        parameter_dictionary = json.loads(open(parameter_path).read())
        seed_file = get_seed_file(path_to_data_folder, parameter_dictionary)
        seed_scores, seed_specialisation = evaluate_model(seed_file)
        seed_fitness = str(np.sum(seed_scores))

        # Log centralised and one-pop once
        if parameter_dictionary["general"]["learning_type"] == "centralised":
            agent_index = "None"
            parameters_to_log = parameter_list + [agent_index] + [seed_fitness]

            if parameter_dictionary["general"]["reward_level"] == "team":
                fitness = str(np.sum(agent_scores))

            elif parameter_dictionary["general"]["reward_level"] == "individual":
                fitness = str(np.mean(agent_scores))

            parameters_to_log += [fitness] + [str(seed_specialisation)] + [str(specialisation)] + [model_path]

            line_to_log = ",".join(parameters_to_log)
            f.write(line_to_log)
            f.write("\n")

        # Log decentralised once for each team-member
        if parameter_dictionary["general"]["learning_type"] == "decentralised" and parameter_dictionary["general"][
            "reward_level"] == "individual":

            for agent_index in range(agent_scores):
                parameters_to_log = parameter_list + [agent_index] + [seed_fitness]
                fitness = str(agent_scores[agent_index])
                parameters_to_log += [fitness] + [str(seed_specialisation)] + [str(specialisation)] + [model_path]

                line_to_log = ",".join(parameters_to_log)
                f.write(line_to_log)
                f.write("\n")

            evaluated_teams += ["_".join(parameter_list)]

    # Close results file
    f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Reevaluate models and save to file')
    parser.add_argument('--data_path', action="store")
    parser.add_argument('--generation', action="store")

    data_path = parser.parse_args().data_path
    generation = parser.parse_args().generation

    create_reevaluated_results(data_path, generation)