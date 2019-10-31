from algorithm.parameters import params
from fitness.base_ff_classes.base_ff import base_ff
from os import path, getcwd, pardir, chdir
import subprocess
import time

import gym

class task_specialisation(base_ff):
    """Calculates fitness according to the function defined in the ARGoS Loop Functions"""

    maximise = True
    dots = None

    def __init__(self):
        # Initialise base fitness function class.
        super().__init__()
        self.env = gym.make('gym_TS:TS-v2')

    def escape_characters(self, escape_string):
        escape_string = escape_string.replace(";","\;")
        return escape_string

    def evaluate(self, ind, **kwargs):
        random_seed = 1  # get from parameters (pass as parameter from TS_NE.py
        simulation_length = 1000
        render = False
        num_trials = 3
        avg_score = 0.0
        GEAgent individual = GEAgent(ind)

        for trial in range(num_trials):
            self.env.seed(random_seed)
            observations = self.env.reset()
            score = 0
            done = False

            for t in range(simulation_length):
                if render:
                    self.env.render()

                robot_actions = [individual.act(observations[i]) for i in range(len(observations))]

                # The environment changes according to all their actions
                observations, reward, done, info = self.env.step(robot_actions, t)
                score += reward

                if done:
                    break

            avg_score += score
            random_seed += 1

        avg_score /= num_trials

        return avg_score
