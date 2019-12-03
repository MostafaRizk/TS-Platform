import gym
import numpy as np

from agents.TinyAgent import TinyAgent
from gym.utils import seeding

import time


class FitnessCalculator:
    def __init__(self, output_selection_method, random_seed=0, simulation_length=1000):
        self.env = gym.make('gym_TS:TS-v2')
        # env = gym.wrappers.Monitor(env, 'video', force = True) # Uncomment to save video
        # Get size of input and output space and creates agent
        self.observation_size = self.env.get_observation_size()
        self.action_size = self.env.get_action_size()

        # Seeding values
        self.random_seed = random_seed
        self.np_random, self.random_seed = seeding.np_random(self.random_seed)

        # Action space uses a separate random number generator so need to set its seed separately
        self.env.action_space.np_random.seed(self.random_seed)

        self.simulation_length = simulation_length

        self.output_selection_method = output_selection_method

    def get_observation_size(self):
        return self.observation_size

    def get_action_size(self):
        return self.action_size

    def get_rng(self):
        return self.np_random

    def calculate_fitness(self, individual, num_trials=5, render=False, learning_method="NE"):
        """
        Calculates fitness of a controller by running a simulation
        :param render:
        :param num_trials:
        :param individual:
        :param learning_method Accepts NE, DQN or GE
        :return:
        """

        #render=True
        average_score = 0
        temp_seed = self.random_seed

        for trial in range(num_trials):
            if learning_method == "NE" and not isinstance(individual, TinyAgent):
                temp_individual = TinyAgent(self.observation_size, self.action_size, self.output_selection_method, temp_seed)
                temp_individual.load_weights(individual)
                individual = temp_individual

            self.env.seed(temp_seed)  # makes fitness deterministic
            observations = self.env.reset()
            if learning_method == "DQN":
                for i in range(len(observations)):
                    observations[i] = np.array(observations[i])
                    observations[i] = np.reshape(observations[i], [1, self.get_observation_size()])
            score = 0
            done = False

            for t in range(self.simulation_length):
                if render:
                    self.env.render()

                # All agents act using same controller.
                robot_actions = [individual.act(observations[i]) for i in range(len(observations))]
                # robot_actions = [env.action_space.sample() for el in range(env.get_num_robots())]

                # The environment changes according to all their actions
                old_observations = observations[:]
                observations, reward, done, info = self.env.step(robot_actions, t)
                if learning_method == "DQN":
                    for i in range(len(observations)):
                        observations[i] = np.array(observations[i])
                        observations[i] = np.reshape(observations[i], [1, self.get_observation_size()])

                score += reward

                if learning_method == "DQN":
                    for i in range(len(robot_actions)):
                        individual.remember(old_observations[i], robot_actions[i], reward, observations[i], done)

                #Display observation/action in readable format
                '''
                actions = ["Forward ", "Backward", "Left    ", "Right   ", "Drop    ", "Pickup  "]
                tile = ["Empty", "Contains thing"]
                location = ["Nest  ", "Cache ", "Slope ", "Source"]
                carrying = ["Not carrying", "Carrying    "]

                sensed_tile = ''
                sensed_location = ''
                sensed_object = ''

                observation = old_observations[0]

                for j in range(len(observation)):
                    if observation[j] == 1:
                        if j == 0:
                            sensed_tile = tile[1]
                        if 1 <= j <= 4:
                            sensed_location = location[j - 1]
                        if j == 5:
                            sensed_object = carrying[1]
                    if observation[j] == 0 and j == 5:
                        sensed_object = carrying[0]
                    if observation[j] == 0 and j == 0:
                        sensed_tile = tile[0]

                print(
                    f'{sensed_tile}     {sensed_location}     {sensed_object}   --->   {actions[robot_actions[0]]}')
                '''

                #time.sleep(1)
                # print(f'Time: {t} || Score: {score}')

                if done:
                    break

            average_score += score
            temp_seed += 1

            if learning_method == "DQN":
                loss = individual.replay()

        if learning_method == "DQN":
            return average_score/num_trials, individual

        return average_score/num_trials

    def calculate_fitness_negation(self, individual, render=False):
        render = True
        return -1*self.calculate_fitness(individual=individual, render=render)
