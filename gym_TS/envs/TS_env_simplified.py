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


class TSSimpEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self, logging=False):
        '''
        Initialise position, speed, gravity and velocity. Also defines action space and observation space. Creates seed?
        :param goal_velocity:
        '''
        self.logging = logging

        self.robot_width = 40
        self.robot_height = 20
        self.res_width = 20
        self.res_height = 30

        self.min_position = 0
        self.max_position = 20
        self.max_speed = 0.07
        self.sensor_range = 1

        self.gravity = 0.0025

        self.cache_start = 4
        self.slope_start = 5
        self.slope_end = 15
        self.slope_angle = 10

        self.behaviour = -1

        self.pickup_delay = 20
        self.pickup_delay_counter = 0
        self.behaviour_delay = 50
        self.behaviour_delay_counter = 0

        self.random_walk_counter = 0
        self.random_walk_length = 10
        self.last_step_was_random = False
        self.last_step_direction = 0

        self.last_behaviour = -1

        self.position = np.random.uniform(
            low=self.min_position, high=self.cache_start)
        self.res_position = np.random.uniform(
            low=self.slope_end, high=self.max_position)

        self.position_map = ["ON_NEST", "ON_CACHE", "ON_SLOPE", "ON_SOURCE"]
        self.behaviour_map = [
            self.phototaxis_step,
            self.antiphototaxis_step,
            self.random_walk_step]
        self.behaviour_map_display = [
            "DO_PHOTOTAXIS",
            "DO_ANTIPHOTOTAXIS",
            "DO_RANDOM_WALK"]
        self.want_map = ["WANT_RESOURCE = False", "WANT_RESOURCE = True"]
        self.has_map = ["HAS_RESOURCE = False", "HAS_RESOURCE = True"]

        self.viewer = None

        # self.action_space = spaces.Discrete(4) #0-Phototaxis,
        # 1-Antiphototaxis, 2-Random walk, 3-Flip want
        # 0-Phototaxis, 1-Antiphototaxis, 2-Flip want
        self.action_space = spaces.Discrete(3)
        # position, want_resource, has_resource, behaviour
        self.observation_space = spaces.Discrete(4)

        self.seed()

    def seed(self, seed=None):
        '''
        Creates random seed and returns it
        :param seed:
        :return:
        '''
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action, time_step):
        '''

        :param action:
        :return: The new state, the reward and whether or not the goal has been achieved
        '''

        # Returns an error if the action is invalid
        assert self.action_space.contains(
            action), "%r (%s) invalid" % (action, type(action))

        reward = 0.0

        area, has_resource, resource_location = self.state

        resource_is_in_range = self.get_robot_location(
            self.res_position) == self.get_robot_location(
            self.position)

        '''
        if self.behaviour_delay_counter == 0:
            self.behaviour = action
            self.last_step_was_random = False
            self.behaviour_delay_counter += 1

        elif self.behaviour_delay_counter < self.behaviour_delay:
            self.behaviour_delay_counter += 1

        else:
            self.behaviour_delay_counter = 0
        '''
        self.behaviour = action

        if self.behaviour != -1:
            self.behaviour_map[self.behaviour]()

        if resource_is_in_range:
            # If resource is in range, keep it in agent's possession
            self.pickup_and_hold_resource()
            has_resource = 1

        self.position = np.clip(
            self.position,
            self.min_position,
            self.max_position)
        self.res_position = np.clip(
            self.res_position,
            self.min_position,
            self.max_position)
        area = self.get_robot_location(self.position)

        done = self.res_position < self.cache_start

        '''
        if done:
            reward += 20000.0
        else:
            reward -= 1.0
        '''

        reward -= 1.0

        self.state = (
            area,
            has_resource,
            self.get_robot_location(
                self.res_position))

        if self.logging:
            self.log_data(time_step)
        return np.array(self.state), reward, done, {}

    def reset(self):
        '''

        :return:
        '''
        self.position = np.random.uniform(
            low=self.min_position, high=self.cache_start)
        self.res_position = np.random.uniform(
            low=self.slope_end, high=self.max_position)

        self.state = np.array([self.get_robot_location(self.position),  # Location
                               0,
                               # Object has_resource (0- HAS_OBJECT=False, 1-
                               # HAS_OBJECT=True)
                               self.get_robot_location(self.res_position)])

        return np.array(self.state)

    def get_state_size(self):
        return 4 + 2 + 4

    def one_hot_encode(self, state):
        classes = [4, 2, 4]
        encoded_state = np.array([])

        for i in range(len(state)):
            encoded_state = np.append(
                encoded_state, to_categorical(
                    state[i], num_classes=classes[i]))

        return encoded_state

    def get_possible_states(self):
        classes = [4, 2, 4]
        possible_states = []

        for i_1 in range(classes[0]):
            for i_2 in range(classes[1]):
                for i_3 in range(classes[2]):
                    possible_states += [np.reshape(self.one_hot_encode(
                        np.array([i_1, i_2, i_3])), [1, self.get_state_size()])]

        return possible_states

    def height_map(self, x):
        '''
        Maps an x coordinate on the track to the height of the track at that point
        :param x:
        :return:
        '''
        if x < self.slope_start:
            return self.slope_start * math.tan(math.radians(self.slope_angle))
        elif x < self.slope_end:
            return x * math.tan(math.radians(self.slope_angle))
        else:
            return self.slope_end * math.tan(math.radians(self.slope_angle))

    def _height(self, xs):
        '''
        Returns height according to the function representing the track
        :param xs:
        :return:
        '''
        if not isinstance(xs, np.ndarray):
            return self.height_map(xs)
        else:
            ys = np.copy(xs)
            for i in range(len(xs)):
                ys[i] = self.height_map(xs[i])
            return ys

    def render(self, mode='human'):
        '''
        Renders the environment and agent(s)
        :param mode:
        :return:
        '''
        screen_width = 900
        screen_height = 600

        world_width = self.max_position - self.min_position
        scale = screen_width / world_width

        if self.viewer is None:
            # Define the geometry of the track
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            xs = np.linspace(self.min_position, self.max_position, 100)
            ys = self._height(xs)
            xys = list(zip((xs - self.min_position) * scale, ys * scale))

            self.track = rendering.make_polyline(xys)
            self.track.set_linewidth(4)
            self.viewer.add_geom(self.track)

            robot_clearance = 10
            res_clearance = 0

            # Create robot
            l, r, t, b = -self.robot_width / 2, \
                self.robot_width / 2, \
                self.robot_height, \
                0
            robot = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            robot.set_color(0, 0, 1)
            robot.add_attr(
                rendering.Transform(
                    translation=(
                        0, robot_clearance)))
            self.robottrans = rendering.Transform()
            robot.add_attr(self.robottrans)
            self.viewer.add_geom(robot)
            frontwheel = rendering.make_circle(self.robot_height / 2.5)
            frontwheel.set_color(.5, .5, .5)
            frontwheel.add_attr(
                rendering.Transform(
                    translation=(
                        self.robot_width / 4,
                        robot_clearance)))
            frontwheel.add_attr(self.robottrans)
            self.viewer.add_geom(frontwheel)
            backwheel = rendering.make_circle(self.robot_height / 2.5)
            backwheel.add_attr(rendering.Transform(
                translation=(-self.robot_width / 4, robot_clearance)))
            backwheel.add_attr(self.robottrans)
            backwheel.set_color(.5, .5, .5)
            self.viewer.add_geom(backwheel)

            # Create resource
            '''
            l, r, t, b = (self.res_position-self.min_position)*scale -self.res_width / 2, \
                         (self.res_position-self.min_position)*scale + self.res_width / 2, \
                         self._height(self.res_position)*scale + self.res_height, \
                         self._height(self.res_position)*scale
            '''
            l, r, t, b = -self.res_width / 2, \
                self.res_width / 2, \
                self.res_height, \
                0
            resource = rendering.FilledPolygon(
                [(l, b), (l, t), (r, t), (r, b)])
            resource.add_attr(
                rendering.Transform(
                    translation=(
                        0, res_clearance)))
            self.restrans = rendering.Transform()
            resource.add_attr(self.restrans)
            self.viewer.add_geom(resource)
            resource.set_color(0, 1, 0)

        # Set position of robot
        robot_pos = self.position
        self.robottrans.set_translation(
            (robot_pos - self.min_position) * scale,
            self._height(robot_pos) * scale)
        #self.robottrans.set_translation(robot_pos * scale, self._height(robot_pos) * scale)

        # Set position of resource
        res_pos = self.res_position
        self.restrans.set_translation(
            (res_pos - self.min_position) * scale,
            self._height(res_pos) * scale)
        #self.restrans.set_translation(res_pos, self._height(res_pos) )
        #self.restrans.set_rotation(math.cos(3 * pos))

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def get_robot_location(self, position):
        '''
        Calculates an integer representing which area the robot is on depending on its position
        :return:
        '''
        #(0 - ON_NEST, 1 - ON_CACHE, 2 - ON_SLOPE, 3 - ON_SOURCE)
        if self.min_position <= position < self.cache_start:
            return 0
        if self.cache_start <= position < self.slope_start:
            return 1
        if self.slope_start <= position < self.slope_end:
            return 2
        else:
            return 3

    def phototaxis_step(self):
        self.position += 1  # 1*self.max_speed

    def antiphototaxis_step(self):
        self.position -= 1  # 1*self.max_speed

    def random_walk_step(self):
        direction_list = [-1, 1]
        # if the last step was a random walk, increment walk counter until it reaches a limit
        # step in the same direction as last step
        if self.last_step_was_random:
            self.random_walk_counter += 1
            if self.random_walk_counter < self.random_walk_length:
                choice = self.last_step_direction
            else:
                self.random_walk_counter = 0
                choice = np.random.randint(low=0, high=2)
        # otherwise reset the walk counter
        else:
            choice = np.random.randint(low=0, high=2)
            self.random_walk_counter += 1

        self.position += direction_list[choice] * self.max_speed

    def pickup_and_hold_resource(self):
        self.res_position = self.position  # copy.deepcopy(self.position)

    def drop_resource(self):
        self.res_position = copy.deepcopy(self.position)

    def slide_cylinder(self):
        self.res_position -= 2

    def log_data(self, time_step):
        print("Step " + str(time_step))
        print(self.position_map[self.state[0]])
        print(self.want_map[self.state[1]])
        print(self.has_map[self.state[2]])
        print(self.behaviour_map_display[self.state[3]])
        print("---------------------------------")

    def get_keys_to_action(self):
        # control with left and right arrow keys
        return {(): 1, (276,): 0, (275,): 2, (275, 276): 1}

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
