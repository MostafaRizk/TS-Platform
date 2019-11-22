import gym

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

    def calculate_fitness(self, individual, render=False):
        """
        Calculates fitness of a controller by running a simulation
        :return:
        """
        if not isinstance(individual, TinyAgent):
            individual = TinyAgent(self.observation_size, self.action_size, self.output_selection_method, self.random_seed)

        self.env.seed(self.random_seed)  # makes fitness deterministic
        observations = self.env.reset()
        score = 0
        done = False

        for t in range(self.simulation_length):
            if render:
                self.env.render()

            # All agents act using same controller.
            robot_actions = [individual.act(observations[i]) for i in range(len(observations))]
            # robot_actions = [env.action_space.sample() for el in range(env.get_num_robots())]

            # The environment changes according to all their actions
            observations, reward, done, info = self.env.step(robot_actions, t)
            score += reward

            #time.sleep(1)
            #print(robot_actions)

            if done:
                break

        return score

    def calculate_fitness_negation(self, individual, render=False):
        #render = True
        return -1*self.calculate_fitness(individual=individual, render=render)
