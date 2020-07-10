from agents.nn_agent import NNAgent
from fitness import FitnessCalculator
from learning.cma import CMALearner

parameter_filename = "default_parameters.json"
fitness_calculator = FitnessCalculator(parameter_filename)

learner = CMALearner(fitness_calculator)
genome, fitness = learner.learn()