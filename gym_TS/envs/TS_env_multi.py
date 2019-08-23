import gym
from gym import error, spaces, utils
from gym.utils import seeding

import math
import numpy as np
import copy

class TSMultiEnv(gym.Env):
    metadata = {
      'render.modes': ['human', 'rgb_array'],
      'video.frames_per_second': 30
    }

    def __init__(self):
        pass

    def seed(self, seed=None):
        '''
        Creates random seed and returns it
        :param seed:
        :return:
        '''
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        pass

    def reset(self):
        pass

    def height_map(self, x):
        pass

    def _height(self, xs):
        pass

    def render(self, mode='human'):
        pass

    def get_robot_location(self, position):
        pass

    def phototaxis_step(self):
        pass

    def antiphototaxis_step(self):
        pass

    def random_walk_step(self):
        pass

    def pickup_step(self):
        pass

    def slide_cylinder(self):
        pass

    def log_data(self):
        pass

    def get_keys_to_action(self):
        return {():1,(276,):0,(275,):2,(275,276):1} #control with left and right arrow keys

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
