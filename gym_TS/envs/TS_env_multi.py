"""
Sets up environment for task specialisation experiments based on the setup by Ferrante et al. 2015
https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1004273

Note: In the code and documentation, the term 'robot' is used to refer to the agents.
"""

import gym
from gym import spaces
from gym.utils import seeding

import numpy as np
import copy

from gym.envs.classic_control import rendering

try:
    from gym.envs.classic_control import rendering
    pass
except:
    raise UserWarning("Could not import rendering")


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
        self.arena_constraints = {"x_min": 0, "x_max": 8, "y_min": 0, "y_max": 12}
        self.nest_size = self.arena_constraints["y_max"] / 12
        self.cache_size = self.nest_size * 2
        self.slope_size = self.nest_size * 6
        self.source_size = self.nest_size * 3
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
        self.sliding_speed = 2  # self.slope_angle / 10

        # Other constants/variables
        self.num_robots = 1
        self.default_num_resources = 5
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
        self.robot_transforms = None
        self.resource_transforms = None

        try:
            self.robot_transforms = [rendering.Transform() for i in range(self.num_robots)]
            self.resource_transforms = [rendering.Transform() for i in range(self.default_num_resources)]
        except:
            pass

        self.robot_positions = [None for i in range(self.num_robots)]
        self.resource_positions = [None for i in range(self.default_num_resources)]

        # Observation space
        #
        # The details are explained in self.generate_robot_observations()
        self.tiles_in_sensing_range = (2*self.sensor_range + 1)**2  # Range=1 -> 9 tiles. Range=2 -> 25 tiles. Robot at the center.
        self.observation_space = spaces.Discrete(self.tiles_in_sensing_range*4 + 5)  # Tiles are onehotencoded

        # Action space
        self.action_space = spaces.Discrete(6)  # 0- Forward, 1- Backward, 2- Left, 3- Right, 4- Pick up, 5- Drop

        # Step variables
        self.behaviour_map = [self.forward_step, self.backward_step, self.left_step, self.right_step]
        self.action_name = ["FORWARD", "BACKWARD", "LEFT", "RIGHT", "PICKUP", "DROP"]
        self.has_resource = [None for i in range(self.num_robots)]

        self.seed_value = None

    def seed(self, seed=None):
        """
        Generates random seed for np.random
        :param seed:
        :return:
        """
        self.np_random, seed = seeding.np_random(seed)
        self.seed_value = seed
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

    def step(self, robot_actions, time_step):
        """
        Given the current state of the environment and the actions the robots are taking, updates the environment accordingly

        :param robot_actions A list of integers representing the action each robot is taking
        :param time_step The current time step within the simulation

        :return A 4-tuple containing: a list containing each robot's observation, the reward at this time step,
        a boolean indicating if the simulation is done, any additional information
        """

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
            #self.state[position[1]][position[0]] = 0
            self.robot_map[position[1]][position[0]] = 0

        # The resources' old positions are wiped out
        for position in self.resource_positions:
            if position != self.dumping_position:
                #self.state[position[1] + self.arena_constraints["y_max"]][position[0]] = 0
                self.resource_map[position[1]][position[0]] = 0

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

            #self.state[robot_collision_positions[i][1]][robot_collision_positions[i][0]] = i + 1
            self.robot_map[robot_collision_positions[i][1]][robot_collision_positions[i][0]] = i + 1

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
                self.delete_resource(self.has_resource[i])
                self.has_resource[i] = None
                reward += 1
                self.spawn_resource()

        # Update the state with the new resource positions
        for i in range(len(self.resource_positions)):
            if self.resource_positions[i] != self.dumping_position:
                #self.state[self.resource_positions[i][1] + self.arena_constraints["y_max"]][self.resource_positions[i][0]] = i + 1
                self.resource_map[self.resource_positions[i][1]][self.resource_positions[i][0]] = i + 1

        # -1 reward at each time step to promote faster runtimes
        #reward -= 1

        # The task is done when all resources are removed from the environment
        #
        if self.current_num_resources == 0:
            done = True
        elif self.current_num_resources < 0:
            raise ValueError("There should never be a negative number of resources")

        self.state = np.concatenate((self.robot_map, self.resource_map), axis=0)  # Fully observable environment
        observations = self.generate_robot_observations()

        #return self.state, reward, done, {}
        return observations, reward, done, {}

    def generate_arena(self):
        """
        Generates 2D matrix representing an empty arena i.e. each tile contains a 0
        :return:
        """

        return [[0 for i in range(self.arena_constraints["x_max"])] for j in range(self.arena_constraints["y_max"])]

    def generate_robot_position(self):
        """
        Generates and returns valid coordinates for a single robot
        :return: x and y coordinates of the robot
        """
        x = self.np_random.randint(low=self.arena_constraints["x_min"], high=self.arena_constraints["x_max"])
        y = self.np_random.randint(low=self.nest_start, high=self.nest_start + self.nest_size)
        return x, y

    def generate_resource_position(self):
        """
        Generates and returns valid coordinates for a single resource
        :return: x and y coordinates of the resource
        """
        x = self.np_random.randint(low=self.arena_constraints["x_min"], high=self.arena_constraints["x_max"])
        y = self.np_random.randint(low=self.source_start, high=self.arena_constraints["y_max"])
        return x, y

    def generate_robot_observations(self):
        """
        Generate a list containing each robot's observation. Each robot observes:

        1. A radius of self.sensor_range around itself. If sensor range is 0, it only checks the tile it is currently
        on. A onehotencoded bit-vector is added to the observation denoting whether the robot detects a blank space,
        another robot (not possible if it's the tile directly underneath), a resource or a wall (also not possible).
        If the radius is greater than 0, then the robot is at the center of a square (side=3 if radius=1,
        side=5 if radius=2 etc). The same readings are added starting from the top left tile and going row by row
        until the bottom right tile.

        2. Where the robot is (Nest, Cache, Slope, Source) also encoded as a bit-vector

        3. Whether or not the robot is carrying a resource (encoded as a single bit)
        :return:
        """

        observations = []
        readable_observations = []

        for j in range(len(self.robot_positions)):
            position = self.robot_positions[j]
            observation = [0] * self.observation_space.n
            readable_observation = []

            # If a tile in the robot's sensing range is:
            # Blank-            observation[k + 0] = 1, otherwise 0
            # Another robot-    observation[k + 1] = 1, otherwise 0
            # A resource-       observation[k + 2] = 1, otherwise 0
            # A wall-           observation[k + 3] = 1, otherwise 0
            # Each tile is represented by 4 indices instead of 1(for purposes of onehotencoding)
            current_x = position[0] - self.sensor_range
            current_y = position[1] + self.sensor_range
            row_progress = 0

            for k in range(self.tiles_in_sensing_range):
                # If coordinate is out of bounds then it is a wall
                if self.arena_constraints["x_max"] <= current_x or current_x < 0 or self.arena_constraints["y_max"] <= current_y or current_y < 0:
                    observation[4*k+3] = 1  # Wall
                    readable_observation += ["Wall"]
                # If coordinate contains a robot and the robot is not this robot
                elif self.robot_map[current_y][current_x] != 0 and (current_x != position[0] or current_y != position[1]):
                    observation[4*k+1] = 1  # Another robot
                    readable_observation += ["Robot"]
                # If coordinate is a resource
                elif self.resource_map[current_y][current_x] != 0 and self.has_resource[j] != self.resource_map[current_y][current_x]-1:
                    observation[4*k+2] = 1  # A resource
                    readable_observation += ["Resource"]
                else:
                    observation[4*k+0] = 1  # Blank space
                    readable_observation += ["Blank"]

                row_progress += 1

                if row_progress >= self.sensor_range*2 + 1:
                    row_progress = 0
                    current_x = position[0] - self.sensor_range
                    current_y -= 1
                else:
                    current_x += 1

            area = self.get_area_from_position(position)
            obs_index = self.tiles_in_sensing_range*4

            # If the area the robot is located in is
            # The nest-     observation[4] = 1, otherwise 0
            # The cache-    observation[5] = 1, otherwise 0
            # The slope-    observation[6] = 1, otherwise 0
            # The source-   observation[7] = 1, otherwise 0
            if area == "NEST":
                observation[obs_index] = 1
            elif area == "CACHE":
                observation[obs_index+1] = 1
            elif area == "SLOPE":
                observation[obs_index+2] = 1
            else:
                observation[obs_index+3] = 1

            readable_observation += [area]

            # If the robot
            # Has a resource-   observation[8] = 1, otherwise 0
            if self.has_resource[j] is not None:
                observation[obs_index+4] = 1
                readable_observation += ["Has"]
            else:
                readable_observation += ["Doesn't have"]

            observations += [np.array(observation)]
            readable_observations += [readable_observation]

        return observations

    def reset(self):
        """
        """

        # Make sure robots and resources will all fit in the environment
        assert self.num_robots < self.arena_constraints[
            "x_max"] * self.nest_size, "Not enough room in the nest for all robots"
        assert self.default_num_resources < self.arena_constraints[
            "x_max"] * self.source_size, "Not enough room in the source for all resources"

        try:
            self.viewer.close()
        except:
            pass

        self.viewer = None

        self.resource_positions = [None for i in range(self.default_num_resources)]

        try:
            self.resource_transforms = [rendering.Transform() for i in range(self.default_num_resources)]
        except:
            pass

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
        '''
        # Places straight line of resources
        for i in range(self.default_num_resources):
            x, y = i, self.arena_constraints["y_max"]-1
            self.resource_map[y][x] = i + 1
            self.resource_positions[i] = (x,y)
        '''

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
        for i in range(len(self.resource_positions)):
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
        if self.viewer is not None:
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
                try:
                    self.resource_transforms += [rendering.Transform()]
                except:
                    pass
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
                if self.resource_map[int(self.source_start) + y][x] == 0:
                    return False

        return True

    def log_data(self, time_step):
        pass

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
