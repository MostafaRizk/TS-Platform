import tinynet
import numpy as np

from gym.utils import seeding


def linear_activation(x):
    y = x
    return y

class TinyAgent:
    def __init__(self, observation_size, action_size, output_selection_method, seed=None,):
        # Hidden layers are arbitrarily added
        self.hidden = []
        self.net_struct = [observation_size, *self.hidden, action_size]

        # Network setup is straightforward (defaults: `act_fn=np.tanh, init_weights=None`)
        #self.net = tinynet.RNN(self.net_struct)

        self.net = tinynet.FFNN(self.net_struct, act_fn=linear_activation)

        self.np_random, seed = seeding.np_random(seed)

        self.output_selection_method = output_selection_method

    def get_weights(self):
        return self.net.get_weights()

    def get_num_weights(self):
        return self.net.nweights

    def load_weights(self, weights=[]):
        if weights == []:
            weights = self.np_random.randn(self.net.nweights)
        self.net.set_weights(weights)

    def act(self, observation):
        if self.output_selection_method == "argmax":
            #print("Using this one")
            action = self.net.activate(observation).argmax()
            return action

        elif self.output_selection_method == "weighted_probability":
            action_list = self.net.activate(observation)
            min_value = action_list.min()
            action_list = [value + np.abs(min_value) for value in action_list]
            sum_actions = np.sum(action_list)
            choice = self.np_random.uniform(0, sum_actions)

            current = 0
            for i in range(len(action_list)):
                current += action_list[i]
                if current > choice:
                    return i



    def save_model(self):
        weights = self.net.get_weights()
        np.save("models/Tiny/weights", weights)
