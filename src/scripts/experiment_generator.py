import json
import sys

import numpy as np

from learning.cma import CMALearner
from learning.learner_parent import Learner
from glob import glob

# Generate CMA experiments
''''''
core_parameter_filename = "../rwg_team_10_agents.json"
parameter_dictionary = json.loads(open(core_parameter_filename).read())

# If using rwg json instead of default_parameters
parameter_dictionary["general"]["algorithm_selected"] = "cma"
parameter_dictionary["algorithm"]["agent_population_size"] = 100
parameter_dictionary["algorithm"]["cma"]["generations"] = 1000

# If individual
if parameter_dictionary["general"]["reward_level"] == "individual":
    environment_name = parameter_dictionary["general"]["environment"]
    parameter_dictionary["environment"][environment_name]["num_agents"] = 10

num_experiments = 30
np_random = np.random.RandomState(1)
g = open("../LIST_cma", "a")

for i in range(num_experiments):
    new_seed = np_random.randint(low=1, high=2**32-1)
    parameter_dictionary["general"]["seed"] = new_seed
    parameters_in_filename = []
    parameters_in_filename += Learner.get_core_params_in_model_name(parameter_dictionary)
    parameters_in_filename += CMALearner.get_additional_params_in_model_name(parameter_dictionary)
    filename = "_".join([str(param) for param in parameters_in_filename]) + ".json"
    f = open("../"+filename, "w")
    dictionary_string = json.dumps(parameter_dictionary, indent=4)
    f.write(dictionary_string)
    f.close()
    g.write(f"python3 experiment.py --parameters {filename}\n")

g.close()


'''
#for every json file except lazy_generalist and default_parameters
json_files = glob(f'../cma_*.json')

for parameter_filename in json_files:
    # load parameter dictionary
    parameter_dictionary = json.loads(open(parameter_filename).read())

    # change to heterogeneous
    parameter_dictionary["general"]["team_type"] = "heterogeneous"
    parameter_filename = parameter_filename.replace("homogeneous", "heterogeneous")

    # write to file
    f = open(parameter_filename, "w")
    dictionary_string = json.dumps(parameter_dictionary, indent=4)
    f.write(dictionary_string)
    f.close()
'''




