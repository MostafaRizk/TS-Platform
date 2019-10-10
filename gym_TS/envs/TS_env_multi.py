"""
Code modified from OpenAI's Mountain car example
"""

import gym
from gym import spaces
from gym.utils import seeding

import math
import numpy as np
import copy
from keras.utils import to_categorical
from gym.envs.classic_control import rendering


class TSMultiEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self, logging=False):
        '''
        Initialises constants and variables for robots, resources and environment
        :param logging:
        '''
        self.logging = logging

        # Robot constants
        self.robot_width = 40
        self.robot_height = 20
        self.max_speed = 1
        self.sensor_range = 1

        # Resource constants
        self.res_width = 20
        self.res_height = 30
        self.sliding_speed = 2

        # Environment dimensions
        self.arena_constraints = {"x_min": 0, "x_max": 4, "y_min": 0, "y_max": 12}
        self.nest_size = self.arena_constraints["y_max"] / 12  # 4/48
        self.cache_size = self.nest_size * 2  # 8/48
        self.slope_size = self.nest_size * 6  # 24/48
        self.source_size = self.nest_size * 3  # 12/48
        self.nest_start = self.arena_constraints["y_min"]
        self.cache_start = self.nest_start + self.nest_size
        self.slope_start = self.cache_start + self.cache_size
        self.source_start = self.slope_start + self.slope_size
        self.num_arena_tiles = self.arena_constraints["x_max"] * self.arena_constraints["y_max"]
        self.slope_angle = 10
        self.gravity = 9.81

        # Other constants
        self.num_robots = 1
        self.num_resources = 1

        self.viewer = None
        self.scale = 50  # Scale for rendering

        self.action_space = spaces.Discrete(3)  # 0- Phototaxis 1- Antiphototaxis 2-Random walk
        self.observation_space = spaces.Discrete(self.num_arena_tiles)  # Every square in the arena

        self.seed()

    def seed(self, seed=None):
        '''
        Generates random seed for np.random
        :param seed:
        :return:
        '''
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action, time_step):
        pass

    def generate_robot_position(self):
        '''
        Generates and returns valid coordinates for a single robot
        :return: x and y coordinates of the robot
        '''
        x = np.random.randint(low=self.arena_constraints["x_min"], high=self.arena_constraints["x_max"])
        y = np.random.randint(low=self.nest_start, high=self.nest_start + self.nest_size)
        return x, y

    def generate_resource_position(self):
        '''
        Generates and returns valid coordinates for a single resource
        :return: x and y coordinates of the resource
        '''
        x = np.random.randint(low=self.arena_constraints["x_min"], high=self.arena_constraints["x_max"])
        y = np.random.randint(low=self.source_start, high=self.arena_constraints["y_max"])
        return x, y

    def reset(self):
        '''
        Creates a new environment with robots and resources placed in acceptable locations
        :return: The state of the environment (a grid containing '-' for a blank space, 'o' for a resource, 'r' for a
        robot carrying nothing and 'R' for a robot carrying a resource
        '''

        # Make sure robots and resources will all fit in the environment
        assert self.num_robots < self.arena_constraints[
            "x_max"] * self.nest_size, "Not enough room in the nest for all robots"
        assert self.num_resources < self.arena_constraints[
            "x_max"] * self.source_size, "Not enough room in the source for all resources"

        # Creates empty state
        self.state = [['-' for i in range(self.arena_constraints["x_max"])] for j in
                      range(self.arena_constraints["y_max"])]  # Empty environment

        # Places all robots
        for i in range(self.num_robots):
            robot_placed = False
            while not robot_placed:
                x, y = self.generate_robot_position()
                if self.state[y][x] != 'r':
                    self.state[y][x] = 'r'
                    robot_placed = True

        # Places all resources
        for i in range(self.num_resources):
            resource_placed = False
            while not resource_placed:
                x, y = self.generate_resource_position()
                if self.state[y][x] != 'o':
                    self.state[y][x] = 'o'
                    resource_placed = True

        return np.array(self.state)

    def get_state_size(self):
        pass

    def one_hot_encode(self, state):
        pass

    def get_possible_states(self):
        pass

    def get_possible_encoded_states(self):
        pass

    def draw_arena_segment(self, top, bottom, rgb_tuple):
        '''
        Helper function that creates the geometry for a segment of the arena. Intended to be used by the viewer

        :param top:
        :param bottom:
        :param rgb_tuple:
        :return: A FilledPolygon object that can be added to the viewer using add_geom
        '''

        l, r, t, b = self.arena_constraints["x_min"] * self.scale, \
                     self.arena_constraints["x_max"] * self.scale, \
                     top * self.scale, \
                     bottom * self.scale
        arena_segment = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
        arena_segment.add_attr(
            rendering.Transform(
                translation=(
                    0, 0)))
        arena_transform = rendering.Transform()
        arena_segment.add_attr(arena_transform)
        arena_segment.set_color(rgb_tuple[0], rgb_tuple[1], rgb_tuple[2])
        return arena_segment

    def draw_grid(self):
        '''
        Helper function that creates the geometry of gridlines to be used by the viewer

        :return: List of grid lines (PolyLine objects) each to be added to the viewer as a geometry object with add_geom
        '''

        grid_lines = []

        # Draw vertical lines
        verticals_list = np.linspace(self.arena_constraints["x_min"] * self.scale,
                                     self.arena_constraints["x_max"] * self.scale, self.arena_constraints["x_max"] + 1)

        for tick in verticals_list:
            xs = np.array([tick for i in range(self.arena_constraints["y_max"] + 1)])
            ys = np.linspace(self.arena_constraints["y_min"] * self.scale,
                             self.arena_constraints["y_max"] * self.scale, self.arena_constraints["y_max"] + 1)
            xys = list(zip(xs, ys))

            line = rendering.make_polyline(xys)
            grid_lines += [line]

        # Draw horizontal lines
        horizontals_list = np.linspace(self.arena_constraints["y_min"] * self.scale,
                                       self.arena_constraints["y_max"] * self.scale,
                                       self.arena_constraints["y_max"] + 1)

        for tick in horizontals_list:
            xs = np.linspace(self.arena_constraints["x_min"] * self.scale,
                             self.arena_constraints["x_max"] * self.scale, self.arena_constraints["x_max"] + 1)
            ys = np.array([tick for i in range(self.arena_constraints["x_max"] + 1)])
            xys = list(zip(xs, ys))

            line = rendering.make_polyline(xys)
            grid_lines += [line]

        return grid_lines

    def render(self, mode='human'):
        '''
        Renders the environment, placing all robots and resources in appropriate positions
        :param mode:
        :return:
        '''

        screen_width = self.arena_constraints["x_max"] * self.scale
        screen_height = self.arena_constraints["y_max"] * self.scale

        if self.viewer is None:
            self.viewer = rendering.Viewer(screen_width, screen_height)
            nest_colour = [0.25, 0.25, 0.25]
            cache_colour = [1, 0, 0]
            slope_colour = [0, 1, 0]
            source_colour = [0, 0, 1]

            # Draw nest
            nest = self.draw_arena_segment(self.nest_size, self.nest_start, nest_colour)
            self.viewer.add_geom(nest)

            # Draw cache
            cache = self.draw_arena_segment(self.cache_start + self.cache_size,
                                            self.cache_start, cache_colour)
            self.viewer.add_geom(cache)

            # Draw slope
            slope = self.draw_arena_segment(self.slope_start + self.slope_size,
                                            self.slope_start, slope_colour)
            self.viewer.add_geom(slope)

            # Draw slope
            source = self.draw_arena_segment(self.source_start + self.source_size,
                                             self.source_start, source_colour)
            self.viewer.add_geom(source)

            grid_lines = self.draw_grid()
            for line in grid_lines:
                self.viewer.add_geom(line)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def get_robot_location(self, position):
        pass

    def phototaxis_step(self):
        pass

    def antiphototaxis_step(self):
        pass

    def random_walk_step(self):
        pass

    def pickup_and_hold_resource(self):
        pass

    def drop_resource(self):
        pass

    def slide_cylinder(self):
        pass

    def log_data(self, time_step):
        pass

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
