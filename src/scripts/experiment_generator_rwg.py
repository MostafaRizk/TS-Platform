import json
import sys
import os
import argparse

import numpy as np

from learning.rwg_centralised import CentralisedRWGLearner
from learning.rwg_fully_centralised import FullyCentralisedRWGLearner
from learning.learner_parent import Learner
from glob import glob


def generate_rwg_experiments(experiment_directory, core_parameter_filename, learning_type, reward_level, list_file_name, num_agents_in_setup, num_seeds_for_team, generator_seed, many_architectures):
    num_agents_in_setup = [int(num) for num in num_agents_in_setup.strip('[]').split(',')]
    num_seeds_for_team = int(num_seeds_for_team)
    generator_seed = int(generator_seed)

    if learning_type == "decentralised":
        raise RuntimeError("No support for decentralised rwg yet")

    for num_agents in num_agents_in_setup:
        parameter_dictionary = json.loads(open(core_parameter_filename).read())
        environment_name = parameter_dictionary["general"]["environment"]
        parameter_dictionary["general"]["reward_level"] = reward_level
        parameter_dictionary["environment"][environment_name]["num_agents"] = num_agents

        #TODO: Modify this for constant arena size
        if environment_name == "slope":
            parameter_dictionary["environment"][environment_name]["arena_width"] = num_agents * 2
            parameter_dictionary["environment"][environment_name]["num_resources"] = num_agents * 2

        parameter_dictionary["algorithm"]["agent_population_size"] *= (num_agents // 2)
        parameter_dictionary["general"]["learning_type"] = learning_type

        #TODO: WARNING! CHANGE WHEN NOT DOING FC
        if learning_type == "fully-centralised":
            default_hidden_units = 8
        else:
            default_hidden_units = parameter_dictionary["agent"]["nn"]["hidden_units_per_layer"]

        num_experiments = num_seeds_for_team
        np_random = np.random.RandomState(generator_seed)
        g = open(f"{experiment_directory}/data/{list_file_name}", "a")

        for i in range(num_experiments):
            new_seed = np_random.randint(low=1, high=2**32-1)
            parameter_dictionary["general"]["seed"] = new_seed

            for architecture in ["rnn", "ffnn"]:
                for bias in ["True", "False"]:
                    for layers in [0, 1, 2]:
                        parameter_dictionary["agent"]["nn"]["architecture"] = architecture
                        parameter_dictionary["agent"]["nn"]["bias"] = bias
                        parameter_dictionary["agent"]["nn"]["hidden_layers"] = layers

                        if layers == 0:
                            parameter_dictionary["agent"]["nn"]["hidden_units_per_layer"] = 0
                        else:
                            parameter_dictionary["agent"]["nn"]["hidden_units_per_layer"] = default_hidden_units

                        parameters_in_filename = []
                        if not many_architectures:
                            parameters_in_filename += Learner.get_core_params_in_model_name(parameter_dictionary)

                            if learning_type == "centralised":
                                parameters_in_filename += CentralisedRWGLearner.get_additional_params_in_model_name(parameter_dictionary)

                            elif learning_type == "fully-centralised":
                                parameters_in_filename += FullyCentralisedRWGLearner.get_additional_params_in_model_name(parameter_dictionary)

                        else:
                            if learning_type == "centralised":
                                parameters_in_filename += ["CTDE"]
                            elif learning_type == "fully-centralised":
                                parameters_in_filename += ["FC"]

                            parameters_in_filename += [architecture, bias, layers]

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
parser.add_argument('--num_seeds_for_team', action="store", dest="num_seeds_for_team")
parser.add_argument('--generator_seed', action="store", dest="generator_seed")
parser.add_argument('--many_architectures', action="store", dest="many_architectures")

experiment_directory = parser.parse_args().experiment_directory
core_parameter_filename = parser.parse_args().core_parameter_filename
learning_type = parser.parse_args().learning_type
reward_level = parser.parse_args().reward_level
list_file_name = parser.parse_args().list_file_name
num_agents_in_setup = parser.parse_args().num_agents_in_setup
num_seeds_for_team = parser.parse_args().num_seeds_for_team
generator_seed = parser.parse_args().generator_seed
many_architectures = parser.parse_args().many_architectures

if many_architectures == "True":
    many_architectures = True

elif many_architectures == "False":
    many_architectures = False

generate_rwg_experiments(experiment_directory, core_parameter_filename, learning_type, reward_level, list_file_name, num_agents_in_setup, num_seeds_for_team, generator_seed, many_architectures)


'''
--experiment_directory /Users/mostafa/Documents/Code/PhD/TS-Platform/results/2021_09_27_rwg_analysis --core_parameter_filename
'''





