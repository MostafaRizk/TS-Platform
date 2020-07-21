import unittest

from agents.lean_networks.RNN_multilayer import RNN_multilayer

# 2-layer (1 hidden, 1 output) RNN with no bias, linear activation
# 2 input units, 4 hidden units, 1 output unit


class RNNMultilayerTest(unittest.TestCase):
    def test_forward(self):

        # 2 inputs, 1 output, no bias, linear activation
        lean_net = RNN_multilayer(N_inputs=2, N_hidden_layers=0, N_outputs=1, use_bias=False, act_fn="linear")
        lean_net.set_weights_by_list([0.2, 0.2, 0.2])
        inputs = [[0, 0],
                  [1, 1],
                  [1, 1]
                  [1, 0]]
        true_outputs = [0, 0.4, 0.48, 0.296]
        calculated_outputs = []

        for input_value in inputs:
            calculated_outputs += [lean_net.forward(input_value)]

        self.assertEqual(calculated_outputs, true_outputs)

        # 2 inputs, 1 hidden (with 1 unit), 1 output, no bias, linear activation
        lean_net = RNN_multilayer(N_inputs=2, N_hidden_layers=1, N_hidden_units=1, N_outputs=1, use_bias=False, act_fn="linear")
        lean_net.set_weights_by_list([0.2, 0.2, 0.2, 0.5, 0.5])
        inputs = [[1, 1],
                  [1, 0],
                  [0, 1]]
        true_outputs = [0.2, 0.32, 0.252]
        calculated_outputs = []

        for input_value in inputs:
            calculated_outputs += [lean_net.forward(input_value)]

        self.assertEqual(calculated_outputs, true_outputs)

        # 2 inputs, 2 hidden layers (2 units each), 1 output unit, no bias, linear activation
        lean_net = RNN_multilayer(N_inputs=2, N_hidden_layers=2, N_hidden_units=2, N_outputs=1, use_bias=False, act_fn="linear")
        lean_net.set_weights_by_list([0.2, 0.2, 0.5, 0.5, 0.5, 0.5, 0.2, 0.2, 0.6, 0.6, 0.4, 0.8, 0.2, 0.3, 0.6, 0.1, 0.2, 0.4, 0.6])
        inputs = [[1, 0],
                  [1, 1]]
        true_outputs = [0.16, 0.692]
        calculated_outputs = []

        for input_value in inputs:
            calculated_outputs += [lean_net.forward(input_value)]

        self.assertEqual(calculated_outputs, true_outputs)






