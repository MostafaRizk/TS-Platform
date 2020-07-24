import numpy as np
import agents.lean_networks.RNN1L

'''
Extension of RNN1L to have multiple layers
'''

class RNN1L:

    def __init__(self, N_inputs, N_outputs, **kwargs):

        act_fn = kwargs.get('act_fn', 'tanh')
        random_dist = kwargs.get('random_dist', 'normal')

        self.N_inputs = N_inputs
        self.N_outputs = N_outputs

        # MUST DO THIS BEFORE INIT_WEIGHTS!
        random_dists = ['normal', 'uniform']
        assert random_dist in random_dists, 'Must supply valid random dist name!'
        self.random_dist = random_dist

        # MUST DO THIS BEFORE INIT_WEIGHTS!
        use_bias_options = [True, False]
        assert use_bias in use_bias_options, 'Must supply True or False to use_bias!'
        self.use_bias = use_bias

        self.init_weights()

        activation_fn_d = {
                    'tanh' : np.tanh,
                    'linear' : lambda x: x,
                    'relu' : lambda x: np.maximum(0, x)
        }
        assert act_fn in activation_fn_d.keys(), 'Must supply valid activation function name!'
        self.act_fn = activation_fn_d[act_fn]



    def reset_state(self):
        # For RNN, set to 0s.
        self.last_output = np.zeros(self.N_outputs)


    def print_NN_matrices(self):

        '''
        For debugging. Print NN weights/shapes.
        '''

        print('N_inputs: ', self.N_inputs)

        for i,w in enumerate(self.weights_matrix):
            print(f'W_{i} shape: {w.shape}')

        print('N_outputs: ', self.N_outputs)

        print('\nWeight matrices:')
        for i,w in enumerate(self.weights_matrix):
            print(f'\nW_{i}:')
            print(w)


    def init_weights(self):
        # Randomly set the weight matrix, which has to be the right size
        # to include a bias and self.last_output term.

        self.last_output = np.zeros(self.N_outputs)

        mat_input_size = self.N_inputs + self.N_outputs
        if self.use_bias:
            mat_input_size += 1

        if self.random_dist == 'normal':
            self.weights_matrix = np.random.randn(self.N_outputs, mat_input_size)
        elif self.random_dist == 'uniform':
            self.weights_matrix = np.random.uniform(-1.0, 1.0, (self.N_outputs, mat_input_size))
        else:
            raise

        self.N_weights = len(self.weights_matrix.flatten())
        self.reset_state()



    def set_weights(self, weights):
        # Directly set the weights matrix.
        self.weights_matrix = weights
        self.reset_state()


    def set_weights_by_list(self, w_list):
        # Handy for setting externally.
        self.weights_matrix = np.array(w_list).reshape(self.weights_matrix.shape)
        self.reset_state()

    def get_num_weights(self):
        return self.N_weights

    def get_weights_as_list(self):
        return self.weights_matrix.flatten()

    def set_random_weights(self):
        # Just calls init_weights, which will randomize them.
        self.init_weights()
        self.reset_state()


    def forward(self, input_vec):

        '''
        Function for giving input to NN, getting output.
        Matrix multiplies the weight matrix at by a concatenated: (the input,
        a bias term of 1.0, self.last_output), then applies the nonlinear
        activation function and sets self.last_output to this.

        What is done with the output (argmax, etc) is up to the Agent class
        to do.
        '''

        x = input_vec
        if self.use_bias:
            x = np.concatenate((x, [1.0], self.last_output))
        else:
            x = np.concatenate((x, self.last_output))
        x = np.dot(self.weights_matrix, x)
        x = self.act_fn(x)
        self.last_output = x
        return x


    def get_weight_sums(self):

        L0 = np.sum(self.weights_matrix)/self.N_weights
        L1 = np.abs(self.weights_matrix).sum()/self.N_weights
        L2 = (self.weights_matrix**2).sum()/self.N_weights

        return {
            'L0' : L0,
            'L1' : L1,
            'L2' : L2
        }






#
