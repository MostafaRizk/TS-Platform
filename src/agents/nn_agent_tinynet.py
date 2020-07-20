import json
import tinynet
import numpy as np
import warnings
from agents.agent import Agent


def linear(x):
    return x


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


activation_dictionary = {"linear": linear, "sigmoid": sigmoid}


class NNAgent(Agent):
    def __init__(self, observation_size, action_size, parameter_filename, genome=None):
        # Load parameters
        if parameter_filename is None:
            raise RuntimeError("No parameter file specified for the neural network")

        self.parameter_dictionary = json.loads(open(parameter_filename).read())

        # Set activation
        key = self.parameter_dictionary['agent']['nn']['activation_function']
        activation_function = activation_dictionary[key]

        # Set hidden layers
        self.hidden = self.parameter_dictionary['agent']['nn']['hidden_units']
        self.net_structure = [observation_size, *self.hidden, action_size]

        # Create neural network and random number generator
        self.net = None

        if self.parameter_dictionary['agent']['nn']['architecture'] == "rnn" and len(self.hidden) == 0:
            self.net = tinynet.RNN(self.net_structure, act_fn=activation_function)

        elif self.parameter_dictionary['agent']['nn']['architecture'] == "rnn" and len(self.hidden) == 1:
            self.net = tinynet.RNN1L(observation_size, self.hidden[0], act_fn=activation_function)

        elif self.parameter_dictionary['agent']['nn']['architecture'] == "ffnn":
            self.net = tinynet.FFNN(self.net_structure, act_fn=activation_function)

        self.np_random = np.random.RandomState(self.parameter_dictionary['general']['seed'])

        # Set weights if possible
        if genome is None:
            warnings.warn("Creating an agent with no genome")
        else:
            self.net.set_weights(genome)

    def get_genome(self):
        if self.parameter_dictionary['agent']['nn']['architecture'] == "rnn" and len(self.hidden) == 1:
            return self.net.weights_matrix
        else:
            return self.net.get_weights()

    def get_num_weights(self):
        if self.parameter_dictionary['agent']['nn']['architecture'] == "rnn" and len(self.hidden) == 1:
            return self.net.nweights()
        else:
            return self.net.nweights

    def act(self, observation):
        activation_values = self.net.activate(observation)
        action = activation_values.argmax()
        return action

    def get_all_activation_values(self, observation):
        return self.net.activate(observation)

    def save_model(self, name):
        weights = self.net.get_weights()
        np.save(name, weights)

    @staticmethod
    def save_given_model(model, filename):
        np.save(filename, model)

    @staticmethod
    def load_model_from_file(filename):
        try:
            return np.load(filename)
        except:
            raise RuntimeError("Could not load model from given filename")

    @staticmethod
    def get_model_file_extension():
        return ".npy"
