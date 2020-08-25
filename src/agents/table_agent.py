import json

from agents.agent import Agent


class TableAgent(Agent):
    def __init__(self, observation_size, action_size, parameter_filename, genome=None, state_dictionary=None):
        # Load parameters
        if parameter_filename is None:
            raise RuntimeError("No parameter file specified for the neural network")

        self.parameter_dictionary = json.loads(open(parameter_filename).read())

        self.observation_size = observation_size
        self.action_size = action_size
        self.genome = genome
        self.state_dictionary = state_dictionary

    def get_genome(self):
        return self.genome

    def act(self, observation):
        pass

    def get_all_activation_values(self, observation):
        return self.net.forward(observation)

    def save_model(self, name):
        pass

    @staticmethod
    def save_given_model(model, filename):
        pass

    @staticmethod
    def load_model_from_file(filename):
        pass

    @staticmethod
    def get_model_file_extension():
        return ".csv"
