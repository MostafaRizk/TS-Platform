import numpy as np
import random
import copy
import json

from collections import deque
from keras import backend as K
from keras.losses import mean_squared_error


class BasicQAgent:
    def __init__(self, possible_states, action_size):
        self.q_table = {}
        self.possible_states = possible_states
        self.action_size = action_size
        self._build_q_table()
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001

    def _build_q_table(self):
        for state in self.possible_states:
            action_values = []

            for i in range(self.action_size):
                action_values += [np.random.rand()]

            self.q_table[str(state)] = np.array(action_values)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        minibatch = []

        try:
            minibatch = random.sample(self.memory, batch_size)
        except ValueError:
            minibatch = copy.deepcopy(self.memory)

        y_predicted = np.zeros([batch_size, self.action_size])
        y_true = np.zeros([batch_size, self.action_size])
        h = 0

        for state, action, reward, next_state, done in minibatch:
            old_qs = copy.deepcopy(self.q_table[str(state)])  # Q values for state

            target = reward

            if not done:
                target = reward + self.gamma * np.amax(self.q_table[str(next_state)])

            target_f = self.q_table[str(state)]
            target_f[action] = target  # q_values = [ blah, blah, target, blah]
            #self.model.fit(state, target_f, epochs=1, verbose=0)

            self.q_table[str(state)][action] += self.learning_rate*target_f[action]

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
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)

        act_values = self.q_table[str(state)]
        return np.argmax(act_values)  # returns action

    def display(self):
        print(self.q_table)

    def save(self, filepath):
        json_dict = {}

        for key in self.q_table:
            json_dict[key] = self.q_table[key].tolist()

        json_string = json.dumps(json_dict)
        f = open(filepath, 'w')
        f.write(json_string)
        f.close()

    def load(self, filepath):
        f = open(filepath)
        json_dict = json.loads(f.read())
        f.close()

        self.q_table = {}
        for key in json_dict:
            self.q_table[key] = np.array(json_dict[key])
