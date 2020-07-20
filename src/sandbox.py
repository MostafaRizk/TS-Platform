import argparse
from fitness import FitnessCalculator
from learning.rwg import RWGLearner

"""
parser = argparse.ArgumentParser(description='Run an experiment')
parser.add_argument('--parameters', action="store", dest="parameter_filename")
parameter_filename = parser.parse_args().parameter_filename

fitness_calculator = FitnessCalculator(parameter_filename)
learner = RWGLearner(fitness_calculator)
genome, fitness = learner.learn()
"""

from agents.lean_networks.RNN_multilayer import RNN_multilayer
net = RNN_multilayer(N_inputs=2, N_outputs=1, N_hidden_layers=2, N_hidden_units=4, use_bias=False, act_fn="linear")
net.forward([1,2])