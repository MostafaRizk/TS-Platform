"""
Sets up environment for task specialisation experiments based on the setup by Ferrante et al. 2015
https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1004273

Note: In the code and documentation, the term 'robot' is used to refer to the agents.
"""

import numpy as np
import copy


class SlopeEnvParent:
    def __init__(self, num_robots=4, num_resources=5, sensor_range=1, slope_angle=20, arena_length=12, arena_width=8,
                 cache_start=1, slope_start=3, source_start=9, upward_cost_factor=3, downward_cost_factor=0.2,
                 carry_factor=1, resource_reward_factor=1000):
        """
        Initialises constants and variables for robots, resources and environment
        :param
        """
        # Environment dimensions
        self.arena_constraints = {"x_min": 0, "x_max": arena_width, "y_min": 0, "y_max": arena_length}
        self.nest_size = cache_start #self.arena_constraints["y_max"] / 12
        self.cache_size = slope_start - cache_start #self.nest_size * 2
        self.slope_size = source_start - slope_start #self.nest_size * 6
        self.source_size = arena_length - source_start #self.nest_size * 3
        self.nest_start = self.arena_constraints["y_min"]
        self.cache_start = cache_start #self.nest_start + self.nest_size
        self.slope_start = slope_start #self.cache_start + self.cache_size
        self.source_start = source_start #self.slope_start + self.slope_size
        self.num_arena_tiles = self.arena_constraints["x_max"] * self.arena_constraints["y_max"]
        self.slope_angle = slope_angle
        self.gravity = 9.81

        # Robot constants
        self.robot_width = 0.8
        self.robot_height = 0.8
        self.max_speed = 1
        self.sensor_range = sensor_range

        # Resource constants
        self.resource_width = 0.6
        self.resource_height = 0.6
        self.sliding_speed = int(self.slope_angle / 10)
        self.base_cost = 1
        self.reward_for_resource = resource_reward_factor*self.base_cost
        self.upward_cost_factor = upward_cost_factor
        self.downward_cost_factor = downward_cost_factor
        self.carry_factor = carry_factor

        # Other constants/variables
        self.num_robots = num_robots
        self.default_num_resources = num_resources
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

        self.robot_positions = [None]*self.num_robots
        self.resource_positions = [None]*self.default_num_resources
        self.resource_carried_by = [[]]*self.default_num_resources

        # Step variables
        self.behaviour_map = [self.forward_step, self.backward_step, self.left_step, self.right_step]
        self.action_name = ["FORWARD", "BACKWARD", "LEFT", "RIGHT", "PICKUP", "DROP"]
        self.has_resource = [None for i in range(self.num_robots)]

        self.seed_value = None

    def seed(self, seed=None):
        pass

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
        pass

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
        pass

    def reset(self):
        pass

    def get_observation_size(self):
        pass

    def get_action_size(self):
        pass

    def draw_arena_segment(self, top, bottom, rgb_tuple):
        pass

    def draw_grid(self):
        pass

    def render(self, mode='human'):
        pass

    def add_resource_to_rendering(self, resource_id):
        pass

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
        resource_id = self.has_resource[robot_id]

        if robot_id not in self.resource_carried_by[resource_id]:
            self.resource_carried_by[resource_id] += [robot_id]

        self.has_resource[robot_id] = None

    def slide_resource(self, resource_id):
        """
        Lets a resource slide. I.e. the position moves towards the nest
        :param resource_id:
        :return:
        """

        new_x = self.resource_positions[resource_id][0]
        new_y = max(self.resource_positions[resource_id][1] - self.sliding_speed,
                    self.cache_start)

        self.resource_positions[resource_id] = (new_x, new_y)

    def spawn_resource(self):
        pass

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
                if self.resource_map[int(self.source_start) + y][x] == 0 and (x, int(self.source_start) + y) not in self.resource_positions:
                    return False

        return True

    def calculate_ferrante_specialisation(self):
        """
        Calculates task specialisation according to Ferrante et al's measure
        :return:
        """
        resources_retrieved_by_many = 0
        total_resources_retrieved = 0

        for i in range(len(self.resource_positions)):
            if self.resource_positions[i] == self.dumping_position:
                total_resources_retrieved += 1

                if len(self.resource_carried_by[i]) > 1:
                    resources_retrieved_by_many += 1

        if total_resources_retrieved != 0:
            #print(f"{resources_retrieved_by_many} / {total_resources_retrieved}")
            return resources_retrieved_by_many/total_resources_retrieved
        else:
            return 0.0

    def log_data(self, time_step):
        pass

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
