import argparse
import json

from fitness import FitnessCalculator
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

elif parameter_dictionary["general"]["algorithm_selected"] == "cma":
    learner = CMALearner(fitness_calculator)
    genome, fitness = learner.learn()



