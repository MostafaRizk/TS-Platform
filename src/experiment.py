import argparse
import copy
import json

from fitness import FitnessCalculator
from learning.learner_parent import Learner
from learning.rwg import RWGLearner
from learning.cma import CMALearner

parser = argparse.ArgumentParser(description='Run an experiment')
parser.add_argument('--parameters', action="store", dest="parameter_filename")
parameter_filename = parser.parse_args().parameter_filename

parameter_dictionary = json.loads(open(parameter_filename).read())
fitness_calculator = FitnessCalculator(parameter_filename)

if parameter_dictionary["general"]["algorithm_selected"] == "rwg":
    learner = RWGLearner(fitness_calculator)
    genome, fitness = learner.learn()

elif parameter_dictionary["general"]["algorithm_selected"] == "cma" or \
        parameter_dictionary["general"]["algorithm_selected"] == "partialcma":
    learner = CMALearner(fitness_calculator)
    genome, fitness = learner.learn()

elif parameter_dictionary["general"]["algorithm_selected"] == "cma_with_seeding":

    # Load default rwg parameters
    rwg_parameter_filename = 'default_rwg_parameters.json'
    rwg_parameter_dictionary = json.loads(open(rwg_parameter_filename).read())

    # Copy general parameters from cma to rwg
    rwg_parameter_dictionary["general"] = copy.deepcopy(parameter_dictionary["general"])
    rwg_parameter_dictionary["general"]["algorithm_selected"] = "rwg"

    # Copy environment parameters from cma to rwg
    rwg_parameter_dictionary["environment"] = copy.deepcopy(parameter_dictionary["environment"])

    if rwg_parameter_dictionary["general"] == "individual":
        environment_name = rwg_parameter_dictionary["general"]["environment"]
        rwg_parameter_dictionary["environment"][environment_name]["num_agents"] = 1

    # Copy agent parameters from cma to rwg
    rwg_parameter_dictionary["agent"] = copy.deepcopy(parameter_dictionary["agent"])

    # Create rwg json file and load to fitness calculator
    parameters_in_name = Learner.get_core_params_in_model_name(rwg_parameter_dictionary)
    parameters_in_name += RWGLearner.get_additional_params_in_model_name(rwg_parameter_dictionary)
    rwg_filename = "_".join([str(param) for param in parameters_in_name]) + ".json"
    f = open(rwg_filename, "w")
    rwg_dictionary_string = json.dumps(rwg_parameter_dictionary, indent=4)
    f.write(rwg_dictionary_string)
    f.close()
    rwg_fitness_calculator = FitnessCalculator(rwg_parameter_filename)

    # Seeding
    learner1 = RWGLearner(rwg_fitness_calculator)
    genome1, fitness1 = learner1.learn()

    # Learning
    learner2 = CMALearner(fitness_calculator)
    genome2, fitness2 = learner2.learn()




