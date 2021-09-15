import os
import json
import numpy as np
import argparse

from glob import glob
from learning.learner_parent import Learner
from evaluation.evaluate_model import evaluate_model
from evaluation.create_results_from_models import get_seed_file
from itertools import combinations


def create_reevaluated_results(path_to_data_folder, generation, episodes, num_agents_to_remove=0, min_team_size=2, max_team_size=None, has_seed_file=False, env="slope", bc_measure=None):
    # TODO: Modify to avoid repetition of other function
    # Get list of models
    model_files = glob(f'{path_to_data_folder}/*cma*_{generation}.npy')  # TODO: Allow different algorithms

    # Create final results file
    if num_agents_to_remove != 0:
        results_file = os.path.join(path_to_data_folder, f'robustness_{num_agents_to_remove}_removed.csv')
    else:
        results_file = os.path.join(path_to_data_folder, f'results_reevaluated_{episodes}ep_{generation}.csv')

    f = open(results_file, 'w')

    # Write header
    # TODO: Get the header using the learner classes? What about 'agents_removed'?
    if env == "slope":
        header = "learning_type,algorithm_selected,team_type,reward_level,agent_type,environment,seed,num_agents,num_resources,sensor_range,sliding_speed,arena_length,arena_width,cache_start,slope_start,source_start,base_cost,upward_cost_factor,downward_cost_factor,carry_factor,resource_reward,episode_length,num_episodes, incremental_rewards,architecture,bias,hidden_layers,hidden_units_per_layer,activation_function,agent_population_size,sigma,generations,tolx,tolfunhist,tolflatfitness,tolfun,agents_removed,seed_fitness,fitness,seed_specialisation,specialisation,participation,total_resources,behaviour_characterisation,trajectory,model_name"
    elif env == "tmaze":
        header = "learning_type,algorithm_selected,team_type,reward_level,agent_type,environment,seed,num_agents,hall_size,start_zone_size,episode_length,num_episodes,architecture,bias,hidden_layers,hidden_units_per_layer,activation_function,agent_population_size,sigma,generations,tolx,tolfunhist,tolflatfitness,tolfun,agents_removed,seed_fitness,fitness,seed_specialisation,specialisation,behaviour_characterisation,model_name"

    f.write(header)
    f.write("\n")

    # List of all evaluated decentralised teams
    # (Multiple members of the same team will be encountered in the loop,
    # this prevents the same team being evaluated multiple times)
    evaluated_teams = []

    combinations_to_remove = {}

    for team_size in range(min_team_size, max_team_size+2, 2):
        if num_agents_to_remove != 0 and num_agents_to_remove < team_size:
            # Uncomment to try all combinations of agents removed
            # combinations_to_remove[team_size] = [list(combo) for combo in list(combinations([i for i in range(team_size)], num_agents_to_remove))]
            combinations_to_remove[team_size] = [[i for i in range(num_agents_to_remove)]]
        else:
            combinations_to_remove[team_size] = [None]

    # Get list of agent_scores for each
    for model_path in model_files:

        learning_type = model_path.split("/")[-1].split("_")[0]

        if learning_type == "centralised" or learning_type == "fully-centralised":
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
            raise RuntimeError("Model prefix must be centralised, decentralised or fully-centralised")

        parameter_path = os.path.join(path_to_data_folder, parameter_filename)
        parameter_dictionary = json.loads(open(parameter_path).read())

        environment = parameter_dictionary["general"]["environment"]
        num_agents = parameter_dictionary["environment"][environment]["num_agents"]

        # If the model has n agents but n or more agents must be removed, don't evaluate that model
        if num_agents_to_remove >= num_agents or num_agents > max_team_size or num_agents < min_team_size:
            continue

        for combo in combinations_to_remove[num_agents]:
            if not episodes:
                results = evaluate_model(model_path=model_path, ids_to_remove=combo, bc_measure=bc_measure)
            else:
                results = evaluate_model(model_path=model_path, episodes=int(episodes), ids_to_remove=combo, bc_measure=bc_measure)

            if has_seed_file:
                seed_file = get_seed_file(path_to_data_folder, parameter_dictionary)

                if not episodes:
                    seed_results = evaluate_model(model_path=seed_file)
                else:
                    seed_results = evaluate_model(model_path=seed_file, episodes=int(episodes))

                #seed_scores = [np.mean(scores) for scores in seed_results['fitness_matrix']]
                #seed_fitness = str(np.sum(seed_scores))
                seed_fitness_to_log = str(seed_results['fitness_matrix']).replace(",", " ")
                seed_spec_to_log = str(seed_results['specialisation_list']).replace(",", " ")

            else:
                seed_fitness_to_log = "N/A"
                seed_spec_to_log = "N/A"

            # Log results
            parameters_to_log = parameter_list + \
                                [str(combo).replace(",", " ")] + \
                                [seed_fitness_to_log] + \
                                [str(results['fitness_matrix']).replace(",", " ")] + \
                                [seed_spec_to_log] + \
                                [str(results['specialisation_list']).replace(",", " ")] + \
                                [str(results['participation_list']).replace(",", " ")] + \
                                [str(results['total_resources']).replace(",", " ")] + \
                                [str(results['behaviour_characterisation_matrix']).replace(",", " ")] + \
                                [str(results['trajectory_matrix']).replace(",", " ")] + \
                                [model_path]

            line_to_log = ",".join(parameters_to_log)
            f.write(line_to_log)
            f.write("\n")

            # Log decentralised once for each team-member
            if parameter_dictionary["general"]["learning_type"] == "decentralised" and parameter_dictionary["general"][
                "reward_level"] == "individual":
                evaluated_teams += ["_".join(parameter_list)]

    # Close results file
    f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Reevaluate models and save to file')
    parser.add_argument('--data_path', action="store")
    parser.add_argument('--generation', action="store")
    parser.add_argument('--episodes', action="store")
    parser.add_argument('--num_agents_to_remove', action="store")
    parser.add_argument('--min_team_size', action="store")
    parser.add_argument('--max_team_size', action="store")
    parser.add_argument('--env', action="store")
    parser.add_argument('--bc_measure', action="store")

    data_path = parser.parse_args().data_path
    generation = parser.parse_args().generation
    episodes = parser.parse_args().episodes
    num_agents_to_remove = parser.parse_args().num_agents_to_remove
    env = parser.parse_args().env
    bc_measure = parser.parse_args().bc_measure

    if num_agents_to_remove:
        num_agents_to_remove = int(num_agents_to_remove)

    min_team_size = parser.parse_args().min_team_size

    if min_team_size:
        min_team_size = int(min_team_size)

    max_team_size = parser.parse_args().max_team_size

    if max_team_size:
        max_team_size = int(max_team_size)

    create_reevaluated_results(data_path, generation, episodes, num_agents_to_remove, min_team_size, max_team_size, env=env, bc_measure=bc_measure)