import tinynet
import numpy as np

from gym.utils import seeding


class TinyAgent:
    def __init__(self, observation_size, action_size, seed=None):
        # Hidden layers are arbitrarily added
        self.hidden = []
        self.net_struct = [observation_size, *self.hidden, action_size]

        # Network setup is straightforward (defaults: `act_fn=np.tanh, init_weights=None`)
        #self.net = tinynet.RNN(self.net_struct)
        self.net = tinynet.FFNN(self.net_struct)

        self.np_random, seed = seeding.np_random(seed)

    def get_weights(self):
        return self.net.get_weights()

    def get_num_weights(self):
        return self.net.nweights

    def load_weights(self, weights=[]):
        if weights == []:
            weights = self.np_random.randn(self.net.nweights)
        self.net.set_weights(weights)

    def act(self, observation):
        action = self.net.activate(observation).argmax()
        return action

    def save_model(self):
        weights = self.net.get_weights()
        np.save("models/Tiny/weights", weights)
