import json
import tinynet
import numpy as np
from agents.Agent import Agent


def linear(x):
    return x


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


activation_dictionary = {"linear": linear, "sigmoid": sigmoid}


class NNAgent(Agent):
    def __init__(self, observation_size, action_size, parameter_filename):
        # Load parameters
        if parameter_filename is None:
            raise RuntimeError("No parameter file specified for the neural network")

        parameter_dictionary = json.loads(open(parameter_filename).read())

        # Set activation
        key = parameter_dictionary['neural_network']['activation_function']
        activation_function = activation_dictionary[key]

        # Set hidden layers
        self.hidden = []
        self.net_structure = [observation_size, *self.hidden, action_size]

        # Create neural network and random number generator
        self.net = tinynet.RNN(self.net_structure, act_fn=activation_function)
        self.np_random = np.random.RandomState(parameter_dictionary['general']['seed'])

    def get_genome(self):
        return self.net.get_weights()

    def get_num_weights(self):
        return self.net.nweights

    def load_weights(self, weights=None):
        if weights is None:
            weights = self.np_random.randn(self.net.nweights)
        self.net.set_weights(weights)

    def act(self, observation):
        return self.net.activate(observation).argmax()

    def get_all_activation_values(self, observation):
        return self.net.activate(observation)

    def save_model(self, name):
        weights = self.net.get_weights()
        np.save(name, weights)
