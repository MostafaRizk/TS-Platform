import json
import numpy as np
import warnings
from agents.agent import Agent
from agents.lean_networks.FFNN_multilayer import FFNN_multilayer
from agents.lean_networks.RNN_multilayer import RNN_multilayer


class NNAgent(Agent):
    def __init__(self, observation_size, action_size, parameter_filename, genome=None):
        # Load parameters
        if parameter_filename is None:
            raise RuntimeError("No parameter file specified for the neural network")

        self.parameter_dictionary = json.loads(open(parameter_filename).read())

        # Set activation
        activation_function = self.parameter_dictionary['agent']['nn']['activation_function']

        # Set hidden layers
        self.num_hidden_layers = self.parameter_dictionary['agent']['nn']['hidden_layers']
        self.num_hidden_units = self.parameter_dictionary['agent']['nn']['hidden_units_per_layer']

        # Set bias
        if self.parameter_dictionary['agent']['nn']['bias'] == "True":
            self.bias = True
        elif self.parameter_dictionary['agent']['nn']['bias'] == "False":
            self.bias = False

        # Create neural network and random number generator
        self.net = None

        if self.parameter_dictionary['agent']['nn']['architecture'] == "rnn":
            self.net = RNN_multilayer(N_inputs=observation_size, N_outputs=action_size, act_fn=activation_function,
                                      use_bias=self.bias, N_hidden_layers=self.num_hidden_layers,
                                      N_hidden_units=self.num_hidden_units)

        elif self.parameter_dictionary['agent']['nn']['architecture'] == "ffnn":
            self.net = FFNN_multilayer(N_inputs=observation_size, N_outputs=action_size, act_fn=activation_function,
                                      use_bias=self.bias, N_hidden_layers=self.num_hidden_layers,
                                      N_hidden_units=self.num_hidden_units)

        self.np_random = np.random.RandomState(self.parameter_dictionary['general']['seed'])

        # Set weights if possible
        if genome is None:
            warnings.warn("Creating an agent with no genome")
        else:
            self.net.set_weights_by_list(genome)

    def get_genome(self):
        return self.net.get_weights_as_list()

    def get_num_weights(self):
        return self.net.get_num_weights()

    def act(self, observation):
        activation_values = self.net.forward(observation)
        action = activation_values.argmax()
        return action

    def get_all_activation_values(self, observation):
        return self.net.forward(observation)

    def save_model(self, name):
        weights = self.get_genome()
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
