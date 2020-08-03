import json
import sys

import numpy as np

from learning.cma import CMALearner
from learning.learner_parent import Learner

core_parameter_filename = "../rnn_no-bias_0HL_linear.json"
parameter_dictionary = json.loads(open(core_parameter_filename).read())
num_experiments = 30
np_random = np.random.RandomState(1)
g = open("../LIST_cma", "w")

for i in range(num_experiments):
    # Set seed
    new_seed = np_random.randint(low=1, high=2**32-1)
    parameter_dictionary["general"]["seed"] = new_seed

    # Adjust cma parameters
    parameter_dictionary["general"]["algorithm_selected"] = "cma"
    parameter_dictionary["algorithm"]["agent_population_size"] = 80
    parameter_dictionary["algorithm"]["cma"]["generations"] = 5000

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
