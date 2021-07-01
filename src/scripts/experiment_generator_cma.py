import json
import sys
import os
import argparse

import numpy as np

from learning.cma_parent import CMALearner
from learning.learner_parent import Learner
from glob import glob


def generate_cma_experiments(experiment_directory, core_parameter_filename, learning_type, reward_level, list_file_name, num_agents_in_setup, holding_constant, num_seeds_for_team, generator_seed):
    num_agents_in_setup = [int(num) for num in num_agents_in_setup.strip('[]').split(',')]
    num_seeds_for_team = int(num_seeds_for_team)
    generator_seed = int(generator_seed)

    if holding_constant == "games_per_run":
        pop_size_for_team = {
                                "centralised": {2: 240, 4: 480, 6: 720, 8: 960, 10: 1200},
                                "decentralised": {2: 120, 4: 120, 6: 120, 8: 120, 10: 120}
                             }
    elif holding_constant == "games_per_learner":
        pop_size_for_team = {
            "centralised": {2: 100, 4: 200, 6: 300, 8: 400, 10: 500},
            "decentralised": {2: 100, 4: 200, 6: 300, 8: 400, 10: 500},
            "fully-centralised": {2: 100, 4: 200, 6: 300, 8: 400, 10: 500}
        }

    elif holding_constant == "games_per_run_2":
        pop_size_for_team = {
            "centralised": {2: 200, 4: 800, 6: 1800}
        }

    for num_agents in num_agents_in_setup:
        parameter_dictionary = json.loads(open(core_parameter_filename).read())
        environment_name = parameter_dictionary["general"]["environment"]
        parameter_dictionary["general"]["reward_level"] = reward_level
        parameter_dictionary["environment"][environment_name]["num_agents"] = num_agents

        # Comment out for constant arena size
        '''if environment_name == "slope":
            parameter_dictionary["environment"][environment_name]["arena_width"] = num_agents * 2
            parameter_dictionary["environment"][environment_name]["num_resources"] = num_agents * 2'''

        parameter_dictionary["algorithm"]["agent_population_size"] = pop_size_for_team[learning_type][num_agents]
        parameter_dictionary["general"]["learning_type"] = learning_type

        num_experiments = num_seeds_for_team
        np_random = np.random.RandomState(generator_seed)
        g = open(f"{experiment_directory}/data/{list_file_name}", "a")

        for i in range(num_experiments):
            new_seed = np_random.randint(low=1, high=2**32-1)
            parameter_dictionary["general"]["seed"] = new_seed
            parameters_in_filename = []
            parameters_in_filename += Learner.get_core_params_in_model_name(parameter_dictionary)
            parameters_in_filename += CMALearner.get_additional_params_in_model_name(parameter_dictionary)
            filename = "_".join([str(param) for param in parameters_in_filename]) + ".json"
            f = open(f"{experiment_directory}/data/{filename}", "w")
            dictionary_string = json.dumps(parameter_dictionary, indent=4)
            f.write(dictionary_string)
            f.close()
            g.write(f"python3 experiment.py --parameters {filename}\n")

        g.close()


parser = argparse.ArgumentParser(description='Generate experiments')
parser.add_argument('--experiment_directory', action="store", dest="experiment_directory")
parser.add_argument('--core_parameter_filename', action="store", dest="core_parameter_filename")
parser.add_argument('--learning_type', action="store", dest="learning_type")
parser.add_argument('--reward_level', action="store", dest="reward_level")
parser.add_argument('--list_file_name', action="store", dest="list_file_name")
parser.add_argument('--num_agents_in_setup', action="store", dest="num_agents_in_setup")
parser.add_argument('--holding_constant', action="store", dest="holding_constant")
parser.add_argument('--num_seeds_for_team', action="store", dest="num_seeds_for_team")
parser.add_argument('--generator_seed', action="store", dest="generator_seed")

experiment_directory = parser.parse_args().experiment_directory
core_parameter_filename = parser.parse_args().core_parameter_filename
learning_type = parser.parse_args().learning_type
reward_level = parser.parse_args().reward_level
list_file_name = parser.parse_args().list_file_name
num_agents_in_setup = parser.parse_args().num_agents_in_setup
holding_constant = parser.parse_args().holding_constant
num_seeds_for_team = parser.parse_args().num_seeds_for_team
generator_seed = parser.parse_args().generator_seed
generate_cma_experiments(experiment_directory, core_parameter_filename, learning_type, reward_level, list_file_name, num_agents_in_setup, holding_constant, num_seeds_for_team, generator_seed)

# Generate rwg experiments

'''og_directory = "/Users/mostafa/Documents/Code/PhD/TS-Platform/results/2021_02_02_specialisation_version_of_oller/experiments"
experiment_directory = "/Users/mostafa/Documents/Code/PhD/TS-Platform/results/2021_02_05_RWG_all_slopes/experiments"
os.chdir(og_directory)
# for each json file in normal directory
json_files = glob(f'*.json')
g = open(f"{experiment_directory}/LIST_rwg", "w")
for speed in [4, 0, 2]:
    for param_file in json_files:
        # Get parameter dict
        parameter_dictionary = json.loads(open(param_file).read())
        parameter_dictionary["environment"]["slope"]["sliding_speed"] = speed
        parameters_in_filename = []
        parameters_in_filename += Learner.get_core_params_in_model_name(parameter_dictionary)
        filename = f"slope_{speed}_" + param_file
        f = open(f"{experiment_directory}/{filename}", "w")
        dictionary_string = json.dumps(parameter_dictionary, indent=4)
        f.write(dictionary_string)
        f.close()
        g.write(f"python3 experiment.py --parameters {filename}\n")

g.close()'''



