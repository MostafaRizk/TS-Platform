import json
import sys
import os

import numpy as np

from learning.cma import CMALearner
from learning.learner_parent import Learner
from glob import glob

# Generate CMA experiments
''''''
experiment_directory = "/Users/mostafa/Documents/Code/PhD/TS-Platform/results/2021_02_12_evo_for_diff_slopes"

core_parameter_filenames = [
    '../default_parameters.json'
    ]

num_agents_in_setup = [2]
pop_size_for_team = [-1, -1, 100, -1, 100, -1, 120, -1, 160, -1, 100]
num_seeds_for_team = [-1, -1, 30, -1, 60, -1, 75, -1, 75, -1, 150]

for i in range(len(core_parameter_filenames)):
    parameter_dictionary = json.loads(open(core_parameter_filenames[i]).read())

    # If using rwg json instead of default_parameters
    parameter_dictionary["general"]["algorithm_selected"] = "cma_with_seeding"
    parameter_dictionary["algorithm"]["cma"]["generations"] = 1000

    environment_name = parameter_dictionary["general"]["environment"]

    # If individual
    if parameter_dictionary["general"]["reward_level"] == "individual":
        parameter_dictionary["environment"][environment_name]["num_agents"] = num_agents_in_setup[i]

    num_agents = parameter_dictionary["environment"][environment_name]["num_agents"]
    parameter_dictionary["algorithm"]["agent_population_size"] = pop_size_for_team[num_agents]

    num_experiments = num_seeds_for_team[num_agents]
    np_random = np.random.RandomState(1)
    g = open(f"{experiment_directory}/experiments/LIST_cma", "a")

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



