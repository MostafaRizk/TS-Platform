import numpy as np
import random
import copy
import json
'''
from collections import deque
from keras import backend as K
from keras.losses import mean_squared_error
'''

from gym.utils import seeding


class BasicQAgent:
    def __init__(self, possible_states, action_size, random_seed, batch_size=32):
        self.q_table = {}
        self.possible_states = possible_states
        self.action_size = action_size
        self._build_q_table()
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.00
        self.epsilon_decay = 0.9999995
        self.learning_rate = 0.001
        self.model_loaded = False
        self.np_random, seed = seeding.np_random(random_seed)
        self.batch_size = batch_size

    def _build_q_table(self):
        for state in self.possible_states:
            action_values = []

            for i in range(self.action_size):
                #action_values += [np.random.rand()]
                action_values += [0]

            self.q_table[str(state)] = np.array(action_values)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        minibatch = []

        indicies = self.np_random.choice(np.arange(len(self.memory)), self.batch_size)
        minibatch = [self.memory[x] for x in indicies]

        y_predicted = np.zeros([self.batch_size, self.action_size])
        y_true = np.zeros([self.batch_size, self.action_size])
        h = 0

        for state, action, reward, next_state, done in minibatch:
            old_qs = copy.deepcopy(self.q_table[str(state)])  # Q values for state

            target = reward

            if not done:
                target = reward + self.gamma * np.amax(self.q_table[str(next_state)])

            target_f = self.q_table[str(state)]
            target_f[action] = target  # q_values = [ blah, blah, target, blah]
            # self.model.fit(state, target_f, epochs=1, verbose=0)

            self.q_table[str(state)][action] += self.learning_rate * target_f[action]

            y_predicted[h] = np.array(old_qs)
            y_true[h] = np.array(target_f)
            h += 1

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # print("y_predicted: " + str(y_predicted))
        # print("y_true: " + str(y_true))
        y_predicted = K.variable(y_predicted)
        y_true = K.variable(y_true)
        loss = np.mean(K.eval(mean_squared_error(y_predicted, y_true)))
        # print("Loss: " + str(loss))
        return loss

    def act(self, state):

        if not self.model_loaded:
            if np.random.rand() <= self.epsilon:
                return random.randrange(self.action_size)

            act_values = self.q_table[str(state)]
            return np.argmax(act_values)  # returns action

        else:
            act_values = self.q_table[str(state)]
            return np.argmax(act_values)  # returns action

    def display(self):
        actions = ["Forward ", "Backward", "Left    ", "Right   ", "Drop    ", "Pickup  "]
        tile = ["Blank", "Robot", "Res  ", "Wall "]
        location = ["Nest  ", "Cache ", "Slope ", "Source"]
        carrying = ["Not carrying", "Carrying    "]

        print(
            f'     Key                  {actions[0]}                  {actions[1]}                  {actions[2]}                  {actions[3]}                  {actions[4]}                  {actions[5]}')

        for key in self.q_table:
            print(
                f'{key}     {self.q_table[key][0]}     {self.q_table[key][1]}     {self.q_table[key][2]}     {self.q_table[key][3]}     {self.q_table[key][4]}     {self.q_table[key][5]}')

        print("\n")

        print(f'Tile      Loc        Carrying       --->   Action')
        print('--------------------------------------------------')

        for key in self.q_table:
            observation = []
            for i in key:
                if i == '0' or i == '1':
                    observation += [int(i)]

            sensed_tile = ''
            sensed_location = ''
            sensed_object = ''

            for j in range(len(observation)):
                if observation[j] == 1:
                    if 0 <= j <= 3:
                        sensed_tile = tile[j]
                    if 4 <= j <= 7:
                        sensed_location = location[j-4]
                    if j == 8:
                        sensed_object = carrying[j-7]
                if observation[j] == 0 and j == 8:
                    sensed_object = carrying[j-8]

            print(f'{sensed_tile}     {sensed_location}     {sensed_object}   --->   {actions[int(np.argmax(self.q_table[key]))]}')

    def save(self, filepath):
        json_dict = {}

        for key in self.q_table:
            json_dict[key] = self.q_table[key].tolist()

        json_string = json.dumps(json_dict)
        f = open(filepath, 'w')
        f.write(json_string)
        f.close()

    def load_model(self, filepath):
        f = open(filepath)
        json_dict = json.loads(f.read())
        f.close()

        self.q_table = {}
        for key in json_dict:
            self.q_table[key] = np.array(json_dict[key])

        self.model_loaded = True
