import tinynet
import numpy as np
import os
from datetime import datetime

from gym.utils import seeding


def linear_activation(x):
    y = x
    return y


class TinyAgent:
    def __init__(self, observation_size, action_size, seed=None,):
        # Hidden layers are arbitrarily added
        self.hidden = []
        self.net_structure = [observation_size, *self.hidden, action_size]

        # Network setup is straightforward (defaults: `act_fn=np.tanh, init_weights=None`)
        self.net = tinynet.RNN(self.net_structure, act_fn=linear_activation)
        self.np_random, seed = seeding.np_random(seed)

    def get_weights(self):
        return self.net.get_weights()

    def get_num_weights(self):
        return self.net.nweights

    def load_weights(self, weights=None):
        if weights is None:
            weights = self.np_random.randn(self.net.nweights)
        self.net.set_weights(weights)

    def act(self, observation):
        return self.net.activate(observation).argmax()

    def save_model(self, name, sub_dir=None):
        weights = self.net.get_weights()

        directory = "models/Tiny/"
        if sub_dir is not None:
            directory += "/" + sub_dir + "/"

        if not os.path.exists(directory):
            os.makedirs(directory)

        #now = datetime.now()
        #timestamp = datetime.timestamp(now)

        #np.save(directory +"weights_"+ name + "_" + str(timestamp), weights)
        np.save(directory + name, weights)
