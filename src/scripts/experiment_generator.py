import json
import sys
import os

import numpy as np

from learning.cma_parent import CMALearner
from learning.learner_parent import Learner
from glob import glob

# Generate CMA experiments
''''''
experiment_directory = "/Users/mostafa/Documents/Code/PhD/TS-Platform/results/2021_02_26_centralised_vs_decentralised_starter"

core_parameter_filename = '../default_parameters.json'

learning_type = "decentralised"
reward_level = "individual"
list_file_name = "LIST_decentralised_4"

num_agents_in_setup = [4]
pop_size_for_team = {2: 100, 4: 100, 6: 120, 8: 160, 10: 100}
num_seeds_for_team = {2: 30, 4: 60, 6: 75, 8: 75, 10: 150}

for num_agents in num_agents_in_setup:
    parameter_dictionary = json.loads(open(core_parameter_filename).read())
    environment_name = parameter_dictionary["general"]["environment"]
    parameter_dictionary["general"]["reward_level"] = reward_level
    parameter_dictionary["environment"][environment_name]["num_agents"] = num_agents
    parameter_dictionary["algorithm"]["agent_population_size"] = pop_size_for_team[num_agents]
    parameter_dictionary["general"]["learning_type"] = learning_type

    num_experiments = num_seeds_for_team[num_agents]
    np_random = np.random.RandomState(1)
    g = open(f"{experiment_directory}/experiments/{list_file_name}", "a")

    for i in range(num_experiments):
        new_seed = np_random.randint(low=1, high=2**32-1)
        parameter_dictionary["general"]["seed"] = new_seed
        parameters_in_filename = []
        parameters_in_filename += Learner.get_core_params_in_model_name(parameter_dictionary)
        parameters_in_filename += CMALearner.get_additional_params_in_model_name(parameter_dictionary)
        filename = "_".join([str(param) for param in parameters_in_filename]) + ".json"
        f = open(f"{experiment_directory}/experiments/{filename}", "w")
        dictionary_string = json.dumps(parameter_dictionary, indent=4)
        f.write(dictionary_string)
        f.close()
        g.write(f"python3 experiment.py --parameters {filename}\n")

    g.close()


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



