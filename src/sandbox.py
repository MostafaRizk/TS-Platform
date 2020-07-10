import argparse

from fitness import FitnessCalculator
from learning.cma import CMALearner

parser = argparse.ArgumentParser(description='Run an experiment')
parser.add_argument('--parameters', action="store", dest="parameter_filename")
parameter_filename = parser.parse_args().parameter_filename
fitness_calculator = FitnessCalculator(parameter_filename)

learner = CMALearner(fitness_calculator)
genome, fitness = learner.learn()