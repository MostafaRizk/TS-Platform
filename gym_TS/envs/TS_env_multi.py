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
        """
        Initialises constants and variables for robots, resources and environment
        :param logging:
        """
        self.logging = logging

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

        # Robot constants
        self.robot_width = 0.8
        self.robot_height = 0.8
        self.max_speed = 1
        self.sensor_range = 1

        # Resource constants
        self.resource_width = 0.6
        self.resource_height = 0.6
        self.sliding_speed = 2

        # Other constants/variables
        self.num_robots = 1
        self.default_num_resources = 3
        self.current_num_resources = self.default_num_resources
        self.latest_resource_id = self.default_num_resources - 1
        self.dumping_position = (-10, -10)

        # Rendering constants
        self.scale = 50  # Scale for rendering
        self.nest_colour = [0.25, 0.25, 0.25]
        self.cache_colour = [0.5, 0.5, 0.5]
        self.slope_colour = [0.5, 0.25, 0.25]
        self.source_colour = [0.25, 0.5, 0.5]
        self.robot_colour = [0, 0, 0.25]
        self.resource_colour = [0, 0.25, 0]

        # Rendering variables
        self.viewer = None
        self.robot_transforms = [rendering.Transform() for i in range(self.num_robots)]
        self.resource_transforms = [rendering.Transform() for i in range(self.default_num_resources)]
        self.robot_positions = [None for i in range(self.num_robots)]
        self.resource_positions = [None for i in range(self.default_num_resources)]

        # Observation space
        # The state has 2 copies of the arena, one with robot locations and one with the resource locations
        # NOTE: To change this, the reset function must also be changed
        #self.observation_space = spaces.Discrete(self.num_arena_tiles * 2)

        # Each robot observes the 8 tiles around it and whether or not it has food
        #self.observation_space = spaces.Discrete(9)

        # Each robot observes directly in front of it, its location and whether or not it has a resource
        self.observation_space = spaces.Discrete(3)

        # Action space
        #self.action_space = spaces.Discrete(3)  # 0- Phototaxis 1- Antiphototaxis 2-Random walk
        self.action_space = spaces.Discrete(6)  # 0- Forward, 1- Backward, 2- Left, 3- Right, 4- Pick up, 5- Drop

        self.seed()

        # Step variables
        #self.behaviour_map = [self.phototaxis_step, self.antiphototaxis_step, self.random_walk_step]
        self.behaviour_map = [self.forward_step, self.backward_step, self.left_step, self.right_step]
        self.action_name = ["FORWARD", "BACKWARD", "LEFT", "RIGHT", "PICKUP", "DROP"]
        self.has_resource = [None for i in range(self.num_robots)]

    def seed(self, seed=None):
        """
        Generates random seed for np.random
        :param seed:
        :return:
        """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_num_robots(self):
        """
        Returns number of robots
        :return: Integer representing number of robots
        """
        return self.num_robots

    def get_default_num_resources(self):
        """
        Returns default number of resources
        :return: Integer representing number of resources
        """
        return self.default_num_resources

    # WARNING: ONLY IMPLEMENTED FOR ONE ROBOT AND ONE RESOURCE
    def get_possible_states(self):
        """
        Generate an array of all possible states
        :return: Array of possible states
        """
        robot_map = self.generate_arena()
        resource_map = self.generate_arena()
        possible_states = []

        for i in range(len(robot_map)):
            for j in range(len(robot_map[0])):
                robot_map[i][j] = 1
                possible_states += [np.concatenate((robot_map, resource_map), axis=0)]
                for k in range(len(resource_map)):
                    for l in range(len(resource_map[0])):
                        resource_map[k][l] = 1
                        possible_states += [np.concatenate((robot_map, resource_map), axis=0)]
                        resource_map[k][l] = 0
                robot_map[i][j] = 0

        return possible_states

    def step(self, robot_actions, time_step):
        # Returns an error if the number of actions is incorrect
        assert len(robot_actions) == self.num_robots, "Incorrect number of actions"

        # Returns an error if any action is invalid
        for action in robot_actions:
            assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        done = False

        reward = 0.0

        # The robots act
        old_robot_positions = copy.deepcopy(self.robot_positions)
        for i in range(len(robot_actions)):
            if robot_actions[i] < 4:
                self.behaviour_map[robot_actions[i]](i)

        # The robots' old positions are wiped out
        for position in old_robot_positions:
            self.state[position[1]][position[0]] = 0

        # The resources' old positions are wiped out
        for position in self.resource_positions:
            if position != self.dumping_position:
                self.state[position[1] + self.arena_constraints["y_max"]][position[0]] = 0

        robot_collision_positions = copy.deepcopy(self.robot_positions)
        # The robots' new positions are updated
        for i in range(len(self.robot_positions)):
            for j in range(len(self.robot_positions)):
                # If the robot's new position is the same as another robot's new position, it stays where it was
                if (self.robot_positions[i] == self.robot_positions[j] or self.robot_positions[i] ==
                    old_robot_positions[j]) and i != j:
                    # If robot i is colliding with robot j's new position, it keeps its old position
                    # If robot i is colliding with robot j's old position, it keeps its old position (this is in case
                    # robot j changes its mind and decides to stay where it is to avoid a collision with a third robot)
                    # But do not update robot i's position in self.robot_positions so that robot j has the chance to
                    # update its position too. Otherwise, robot i will back off but robot j will continue and robots
                    # with higher indices will have an advantage
                    robot_collision_positions[i] = old_robot_positions[i]

            # If robot i's new position is the same as resource j and it is carrying a resource and that resource
            # is not resource j, then stay at the previous position
            # i.e Robots should only collide with resources to pick them up and they can only hold one resource at a
            # time
            for j in range(len(self.resource_positions)):
                if self.robot_positions[i] == self.resource_positions[j]:
                    if self.has_resource[i] is not None and self.has_resource != j:
                        robot_collision_positions[i] = old_robot_positions[i]

            self.state[robot_collision_positions[i][1]][robot_collision_positions[i][0]] = i + 1

        self.robot_positions = robot_collision_positions

        for i in range(len(self.resource_positions)):
            for j in range(len(old_robot_positions)):
                # Ensure that a resource that was at the same location as a robot in the last time step moves with the robot
                # (i.e. the robot holds onto the resource). Also ensure that robot picks it up (unless it was just dropped)
                if self.resource_positions[i] == old_robot_positions[j]:
                    # If a robot currently has a resource and is doing a drop action, drop the resource
                    # If a robot currently has a resource and is NOT doing a drop action, hold onto it
                    if self.has_resource[j] == i:
                        if self.action_name[robot_actions[j]] == "DROP":
                            self.drop_resource(j)
                        else:
                            self.pickup_or_hold_resource(j, i)

                # Ensure that a resource that is at the same location as a robot now gets picked up if the robot is
                # doing a pickup action
                if self.resource_positions[i] == self.robot_positions[j]:
                    if self.has_resource[j] is None:
                        if self.action_name[robot_actions[j]] == "PICKUP":
                            self.pickup_or_hold_resource(j, i)


            # If a resource is on the slope and not in the possession of a robot, it slides
            if self.resource_positions[i] != self.dumping_position and \
                    self.get_area_from_position(self.resource_positions[i]) == "SLOPE" and i not in self.has_resource:
                self.slide_resource(i)

        # If a robot has returned a resource to the nest a new one spawns (sometimes the robot is rewarded)
        for i in range(self.num_robots):
            if self.get_area_from_position(self.robot_positions[i]) == "NEST" and self.has_resource[i] is not None:
                self.resource_positions[self.has_resource[i]] = self.dumping_position
                self.has_resource[i] = None
                self.current_num_resources -= 1
                reward += 1
                self.spawn_resource()

        # Update the state with the new resource positions
        for i in range(len(self.resource_positions)):
            if self.resource_positions[i] != self.dumping_position:
                self.state[self.resource_positions[i][1] + self.arena_constraints["y_max"]][self.resource_positions[i][0]] = i + 1

        # -1 reward at each time step to promote faster runtimes
        #reward -= 1

        # The task is done when all resources are removed from the environment
        if self.current_num_resources == 0:
            done = True
        elif self.current_num_resources < 0:
            raise ValueError("There should never be a negative number of resources")

        observations = self.generate_robot_observations()

        #return self.state, reward, done, {}
        return observations, reward, done, {}

    def generate_arena(self):
        """
        Generates empty arena where each tile contains a 0
        :return:
        """

        return [[0 for i in range(self.arena_constraints["x_max"])] for j in range(self.arena_constraints["y_max"])]

    def generate_robot_position(self):
        """
        Generates and returns valid coordinates for a single robot
        :return: x and y coordinates of the robot
        """
        x = np.random.randint(low=self.arena_constraints["x_min"], high=self.arena_constraints["x_max"])
        y = np.random.randint(low=self.nest_start, high=self.nest_start + self.nest_size)
        return x, y

    def generate_resource_position(self):
        """
        Generates and returns valid coordinates for a single resource
        :return: x and y coordinates of the resource
        """
        x = np.random.randint(low=self.arena_constraints["x_min"], high=self.arena_constraints["x_max"])
        y = np.random.randint(low=self.source_start, high=self.arena_constraints["y_max"])
        return x, y

    def generate_robot_observations(self):
        """
        Generate a list of observations made by each robot
        :return: An np array of observations
        """

        '''
        adjacent_positions_x = [-1, 0, 1, 0, 1, 0, -1, -1]
        adjacent_positions_y = [1, 1, 1, 1, -1, -1, -1, 0]

        observations = []

        for j in range(len(self.robot_positions)):
            position = self.robot_positions[j]
            observation = []

            for i in range(len(adjacent_positions_x)):
                adjacent_position = (position[0] + adjacent_positions_x[i], position[1] + adjacent_positions_y[i])

                if self.arena_constraints["y_max"] <= adjacent_position[1] or \
                        adjacent_position[1] < 0 or \
                        self.arena_constraints["x_max"] <= adjacent_position[0] or \
                        adjacent_position[0] < 0:
                    observation += ["W"]  # Wall

                elif self.robot_map[adjacent_position[1]][adjacent_position[0]] != 0:
                    observation += ["R"]  # Robot

                elif self.resource_map[adjacent_position[1]][adjacent_position[0]] != 0:
                    observation += ["O"]  # Resource

                else:
                    area = self.get_area_from_position(adjacent_position)
                    if area == "SLOPE":
                        observation += ["L"]
                    else:
                        observation += [area[0]]
                    #observation += ["B"]


            if self.has_resource[j]:
                observation += ["F"]
            else:
                observation += ["f"]

            observations += [np.array(observation)]

        return observations
        '''

        observations = []

        for j in range(len(self.robot_positions)):
            position = self.robot_positions[j]
            observation = [None, None, None]

            front = (position[0], position[1] + 1)

            # observation[0] is the space in front of the robot
            # 0- Blank, 1- Another robot, 2- A resource, 3- A wall
            if self.arena_constraints["y_max"] <= front[1] or \
                    front[1] < 0:
                observation[0] = [3]
            elif self.robot_map[front[1]][front[0]] != 0:
                observation[0] = [1]
            elif self.resource_map[front[1]][front[0]] != 0:
                observation[0] = [2]
            else:
                observation[0] = [0]

            area = self.get_area_from_position(position)

            # observation[1] is the area the robot is located in
            # 0- Nest, 1- Cache, 2- Slope, 3- Source
            if area == "NEST":
                observation[1] = [0]
            elif area == "CACHE":
                observation[1] = [1]
            elif area == "SLOPE":
                observation[1] = [2]
            else:
                observation[1] = [3]

            # observation[2] is whether or not the robot has a resource
            # 0- Has no resource, 1- Has resource
            if self.has_resource[j]:
                observation[2] = [1]
            else:
                observation[2] = [0]

            observations += [np.array(observation)]

        return observations

    def get_observation_categories(self):
        """
        Returns categories of all possible observations for use in onehotencoding
        :return: List of cateogires
        """

        #return [["B", "O", "R", "W"] for i in range(self.get_observation_size())]
        return [["C", "F", "L", "N", "O", "R", "S", "W", "f"] for i in range(self.get_observation_size())]

    def reset(self):
        """
        Creates a new environment with robots and resources placed in acceptable locations.
        NOTE: The environment is a flipped version of the state matrix. That is, 0 is the first row of the matrix but
        the bottom of the rendered environment. Additionally, the x,y coordinate of an item in the environment is at
        location [y][x] in the matrix.
        :return: The state of the environment. The state is two concatenated matrices, each representing the
        environment. The first matrix is all 0s except the positions of the robots (robot 0 is 1, robot 1 is 2 etc) and
        the second matrix is all 0s except the positions of the resources.

        Example:
        The environment has y_max=12 and x_max=4
        This is the state representation
        [[0 1 0 0]<--- This is the robot
         [0 0 0 0]
         [0 0 0 0]
         [0 0 0 0]
         [0 0 0 0]
         [0 0 0 0]
         [0 0 0 0]
         [0 0 0 0]
         [0 0 0 0]
         [0 0 0 0]
         [0 0 0 0]
         [0 0 0 0]
         [0 0 0 0]<--- Second matrix starts here
         [0 0 0 0]
         [0 0 0 0]
         [0 0 0 0]
         [0 0 0 0]
         [0 0 0 0]
         [0 0 0 0]
         [0 0 0 0]
         [0 0 0 0]
         [1 0 0 0]<--- This is the resource
         [0 0 0 0]
         [0 0 0 0]]

         This is the arena that would be rendered (R is the robot and S is the resource)
         0 0 0 0
         0 0 0 0
         S 0 0 0
         0 0 0 0
         0 0 0 0
         0 0 0 0
         0 0 0 0
         0 0 0 0
         0 0 0 0
         0 R 0 0

         The robot is at coordinate 1,0 but this would be [0][1] in the first matrix in the state representation
        """

        # Make sure robots and resources will all fit in the environment
        assert self.num_robots < self.arena_constraints[
            "x_max"] * self.nest_size, "Not enough room in the nest for all robots"
        assert self.default_num_resources < self.arena_constraints[
            "x_max"] * self.source_size, "Not enough room in the source for all resources"

        self.resource_positions = [None for i in range(self.default_num_resources)]
        self.resource_transforms = [rendering.Transform() for i in range(self.default_num_resources)]
        self.latest_resource_id = self.default_num_resources - 1


        # Creates empty state
        self.robot_map = self.generate_arena()  # Empty robot map

        self.resource_map = self.generate_arena()  # Empty resource map

        # Places all robots
        for i in range(self.num_robots):
            robot_placed = False
            while not robot_placed:
                x, y = self.generate_robot_position()
                if self.robot_map[y][x] == 0:
                    self.robot_map[y][x] = i + 1
                    self.robot_positions[i] = (x, y)
                    robot_placed = True

        # Places all resources
        for i in range(self.default_num_resources):
            resource_placed = False
            while not resource_placed:
                x, y = self.generate_resource_position()
                if self.resource_map[y][x] == 0:
                    self.resource_map[y][x] = i + 1
                    self.resource_positions[i] = (x, y)
                    resource_placed = True

        # NOTE: To change this, must also change the observation space in __init__
        self.state = np.concatenate((self.robot_map, self.resource_map), axis=0)

        # Reset variables that were changed during runtime
        self.has_resource = [None for i in range(self.num_robots)]
        self.current_num_resources = self.default_num_resources

        #return np.array(self.state)
        return self.generate_robot_observations()

    def get_observation_size(self):
        return self.observation_space.n

    def get_action_size(self):
        return self.action_space.n

    def draw_arena_segment(self, top, bottom, rgb_tuple):
        """
        Helper function that creates the geometry for a segment of the arena. Intended to be used by the viewer

        :param top:
        :param bottom:
        :param rgb_tuple:
        :return: A FilledPolygon object that can be added to the viewer using add_geom
        """

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
        """
        Helper function that creates the geometry of gridlines to be used by the viewer

        :return: List of grid lines (PolyLine objects) each to be added to the viewer as a geometry object with add_geom
        """

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
        """
        Renders the environment, placing all robots and resources in appropriate positions
        :param mode:
        :return:
        """

        screen_width = self.arena_constraints["x_max"] * self.scale
        screen_height = self.arena_constraints["y_max"] * self.scale

        if self.viewer is None:
            self.viewer = rendering.Viewer(screen_width, screen_height)

            # Draw nest
            nest = self.draw_arena_segment(self.nest_size, self.nest_start, self.nest_colour)
            self.viewer.add_geom(nest)

            # Draw cache
            cache = self.draw_arena_segment(self.cache_start + self.cache_size,
                                            self.cache_start, self.cache_colour)
            self.viewer.add_geom(cache)

            # Draw slope
            slope = self.draw_arena_segment(self.slope_start + self.slope_size,
                                            self.slope_start, self.slope_colour)
            self.viewer.add_geom(slope)

            # Draw slope
            source = self.draw_arena_segment(self.source_start + self.source_size,
                                             self.source_start, self.source_colour)
            self.viewer.add_geom(source)

            # Draw grid
            grid_lines = self.draw_grid()
            for line in grid_lines:
                self.viewer.add_geom(line)

            # Draw robot(s)
            for i in range(self.num_robots):
                robot = rendering.make_circle(self.robot_width / 2 * self.scale)
                robot.set_color(self.robot_colour[0], self.robot_colour[1], self.robot_colour[2])
                robot.add_attr(
                    rendering.Transform(
                        translation=(
                            0,
                            0)))
                robot.add_attr(self.robot_transforms[i])
                self.viewer.add_geom(robot)

            # Draw resource(s)
            for i in range(self.default_num_resources):
                resource = rendering.make_circle(self.resource_width / 2 * self.scale)
                resource.set_color(self.resource_colour[0], self.resource_colour[1], self.resource_colour[2])
                resource.add_attr(
                    rendering.Transform(
                        translation=(
                            0,
                            0)))
                resource.add_attr(self.resource_transforms[i])
                self.viewer.add_geom(resource)

        # Set position of robot(s)
        for i in range(self.num_robots):
            self.robot_transforms[i].set_translation(
                (self.robot_positions[i][0] - self.arena_constraints["x_min"] + 0.5) * self.scale,
                (self.robot_positions[i][1] - self.arena_constraints["y_min"] + 0.5) * self.scale)

        # Set position of resource(s)
        for i in range(self.current_num_resources):
            self.resource_transforms[i].set_translation(
                (self.resource_positions[i][0] - self.arena_constraints["x_min"] + 0.5) * self.scale,
                (self.resource_positions[i][1] - self.arena_constraints["y_min"] + 0.5) * self.scale)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def add_resource_to_rendering(self, resource_id):
        resource = rendering.make_circle(self.resource_width / 2 * self.scale)
        resource.set_color(self.resource_colour[0], self.resource_colour[1], self.resource_colour[2])
        resource.add_attr(
            rendering.Transform(
                translation=(
                    0,
                    0)))
        resource.add_attr(self.resource_transforms[resource_id])
        self.viewer.add_geom(resource)

    def get_area_from_position(self, position):
        """
        Gets the area of the environment corresponding to a position i.e. NEST, SOURCE etc
        :param position:
        :return: A string representing the area's name in uppercase
        """

        x = position[0]
        y = position[1]

        if x < self.arena_constraints["x_min"] or x > self.arena_constraints["x_max"]:
            raise ValueError("x position is not valid")

        if self.nest_start <= y < self.cache_start:
            return "NEST"
        elif self.cache_start <= y < self.slope_start:
            return "CACHE"
        elif self.slope_start <= y < self.source_start:
            return "SLOPE"
        elif self.source_start <= y < self.arena_constraints["y_max"]:
            return "SOURCE"
        else:
            raise ValueError("y position is not valid")

    def phototaxis_step(self, robot_id):
        """
        Robot with id robot_id moves one step forward i.e. up in the y direction
        :param robot_id: Index of the robot in self.robot_positions
        :return:
        """
        self.robot_positions[robot_id] = (
            self.robot_positions[robot_id][0],
            np.clip(self.robot_positions[robot_id][1] + 1, self.arena_constraints["y_min"],
                    self.arena_constraints["y_max"] - 1))

    def antiphototaxis_step(self, robot_id):
        """
        Robot with id robot_id moves one step back i.e. down in the y direction
        :param robot_id: Index of the robot in self.robot_positions
        :return:
        """
        self.robot_positions[robot_id] = (
            self.robot_positions[robot_id][0],
            np.clip(self.robot_positions[robot_id][1] - 1, self.arena_constraints["y_min"],
                    self.arena_constraints["y_max"] - 1))

    def random_walk_step(self, robot_id):
        """
        Robot with id robot_id moves randomly to an adjacent tile
        :param robot_id: Index of the robot in self.robot_positions
        :return:
        """
        x_movement = np.random.randint(low=-1, high=2)
        y_movement = np.random.randint(low=-1, high=2)

        self.robot_positions[robot_id] = (
            np.clip(self.robot_positions[robot_id][0] + x_movement, self.arena_constraints["x_min"],
                    self.arena_constraints["x_max"] - 1),
            np.clip(self.robot_positions[robot_id][1] + y_movement, self.arena_constraints["y_min"],
                    self.arena_constraints["y_max"] - 1))

    def forward_step(self, robot_id):
        """
        Robot with id robot_id moves one step forward i.e. up in the y direction
        :param robot_id: Index of the robot in self.robot_positions
        :return:
        """
        self.robot_positions[robot_id] = (
            self.robot_positions[robot_id][0],
            np.clip(self.robot_positions[robot_id][1] + 1, self.arena_constraints["y_min"],
                    self.arena_constraints["y_max"] - 1))

    def backward_step(self, robot_id):
        """
        Robot with id robot_id moves one step back i.e. down in the y direction
        :param robot_id: Index of the robot in self.robot_positions
        :return:
        """
        self.robot_positions[robot_id] = (
            self.robot_positions[robot_id][0],
            np.clip(self.robot_positions[robot_id][1] - 1, self.arena_constraints["y_min"],
                    self.arena_constraints["y_max"] - 1))

    def left_step(self, robot_id):
        """
        Robot with id robot_id moves one step to the left
        :param robot_id: Index of the robot in self.robot_positions
        :return:
        """
        self.robot_positions[robot_id] = (
            np.clip(self.robot_positions[robot_id][0] - 1, self.arena_constraints["x_min"],
                    self.arena_constraints["x_max"] - 1),
            self.robot_positions[robot_id][1])

    def right_step(self, robot_id):
        """
        Robot with id robot_id moves one step to the right
        :param robot_id: Index of the robot in self.robot_positions
        :return:
        """
        self.robot_positions[robot_id] = (
            np.clip(self.robot_positions[robot_id][0] + 1, self.arena_constraints["x_min"],
                    self.arena_constraints["x_max"] - 1),
            self.robot_positions[robot_id][1])

    def pickup_or_hold_resource(self, robot_id, resource_id):
        """
        Lets a robot pick up or hold a resource. I.e. the position of the resource is updated to match that of the
        robot that has picked it up or is holding it. Ensures that the robot is marked as having that resource.
        :param robot_id: Index of the robot in self.robot_positions
        :param resource_id: Index of the resource in self.resource_positions
        :return:
        """
        self.resource_positions[resource_id] = self.robot_positions[robot_id]
        self.has_resource[robot_id] = resource_id

    def drop_resource(self, robot_id):
        """
        Lets a robot drop a resource. I.e. Ensures that the robot is marked as no longer having that resource.
        :param robot_id: Index of the robot in self.robot_positions
        :return:
        """
        self.has_resource[robot_id] = None

    def slide_resource(self, resource_id):
        """
        Lets a resource slide. I.e. the position moves towards the nest
        :param resource_id:
        :return:
        """
        self.resource_positions[resource_id] = (self.resource_positions[resource_id][0], self.resource_positions[resource_id][1] - self.sliding_speed)

    def spawn_resource(self):
        """
        Spawn a new resource in the source area if it is possible to do so
        :return: x,y coordinate of new resource if successful. None otherwise
        """
        # Places all resources
        resource_placed = False

        # If there is no space to spawn new resources, don't spawn
        if self.source_is_full():
            return None

        while not resource_placed:
            x, y = self.generate_resource_position()
            if self.resource_map[y][x] == 0:
                self.resource_map[y][x] = self.latest_resource_id + 1
                self.latest_resource_id += 1
                self.resource_positions += [(x, y)]
                # self.resource_positions[self.latest_resource_id] = (x, y)
                self.resource_transforms += [rendering.Transform()]
                resource_placed = True
                self.current_num_resources += 1
                self.add_resource_to_rendering(self.latest_resource_id)
                return x, y

    def delete_resource(self, resource_id):
        """
        Sends a resource to the dumping position and decrements the resource count
        :param resource_id:
        :return:
        """
        self.resource_positions[resource_id] = self.dumping_position
        self.current_num_resources -= 1

    def source_is_full(self):
        """
        Determines if the source area has a resource at every grid position and is thus "full"
        :return: True if full, False otherwise
        """
        for y in range(int(self.source_size)):
            for x in range(self.arena_constraints["x_max"]):
                if self.resource_map[y][x] == 0:
                    return False

        return True

    def log_data(self, time_step):
        pass

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
