import json
import numpy as np
import copy

# Put this import in a try/except because otherwise, the cluster throws an error
try:
    from helpers import rendering

except:
    pass


class TMazeEnv:
    def __init__(self, parameter_filename=None):
        if parameter_filename is None:
            raise RuntimeError("No parameter file specified for the environment")

        parameter_dictionary = json.loads(open(parameter_filename).read())

        self.seed_value = parameter_dictionary['general']['seed']
        self.np_random = np.random.RandomState(self.seed_value)

        # Environment dimensions
        self.hall_size = parameter_dictionary["environment"]["tmaze"]["hall_size"]
        self.start_zone_size = parameter_dictionary["environment"]["tmaze"]["start_zone_size"]

        if self.start_zone_size % 2 == 0:
            self.offset = 0
        else:
            self.offset = 1

        self.arena_constraints = {"x_min": -(self.start_zone_size//2),
                                  "x_max": (self.start_zone_size//2) + self.offset + self.hall_size,
                                  "y_min": -(self.start_zone_size//2) - self.hall_size,
                                  "y_max": (self.start_zone_size//2) + self.offset + self.hall_size}

        self.obstacle_generation_constraints = {"UP": {"x_min": self.arena_constraints["x_min"],
                                                       "x_max": self.arena_constraints["x_min"] + self.start_zone_size,
                                                       "y_min": self.arena_constraints["y_min"] + self.hall_size + self.start_zone_size,
                                                       "y_max": self.arena_constraints["y_max"]},

                                                "DOWN": {"x_min": self.arena_constraints["x_min"],
                                                         "x_max": self.arena_constraints["x_min"] + self.start_zone_size,
                                                         "y_min": self.arena_constraints["y_min"],
                                                         "y_max": self.arena_constraints["y_min"] + self.hall_size},

                                                "RIGHT": {
                                                    "x_min": self.arena_constraints["x_min"] + self.start_zone_size,
                                                    "x_max": self.arena_constraints["x_max"],
                                                    "y_min": self.arena_constraints["y_min"] + self.hall_size,
                                                    "y_max": self.arena_constraints["y_min"] + self.hall_size + self.start_zone_size}}

        self.total_width = self.start_zone_size + self.hall_size
        self.total_height = self.start_zone_size + self.hall_size + self.hall_size

        self.x_shift = abs(self.arena_constraints["x_min"])
        self.y_shift = abs(self.arena_constraints["y_min"])

        self.num_obstacles_per_hall = parameter_dictionary["environment"]["tmaze"]["num_obstacles_per_hall"]
        self.obstacle_coordinates = {}

        # Place obstacles in each hallway
        for i, hall in enumerate(["UP", "DOWN", "RIGHT"]):
            for j in range(self.num_obstacles_per_hall):
                obstacle_placed = False

                while not obstacle_placed:
                    candidate_position = self.generate_obstacle_position(hall)

                    # Make sure there isn't already an obstacle in that position
                    if candidate_position not in self.obstacle_coordinates:
                        self.obstacle_coordinates[candidate_position] = True
                        obstacle_placed = True

        # Constants and variables
        self.agent_width = 0.8
        self.obstacle_width = 1.0
        self.num_agents = parameter_dictionary["environment"]["tmaze"]["num_agents"]
        self.episode_length = parameter_dictionary["environment"]["tmaze"]["num_episodes"]
        self.specialised_actions = 0
        self.total_rewarded_actions = 0
        self.num_pairs = None

        # Rendering constants
        self.scale = 40
        self.start_zone_colour = [0.5, 0.5, 0.5]
        self.hall_colour = [0.25, 0.5, 0.5]
        self.agent_colour = [0, 0, 0.25]
        self.obstacle_colour = [0.0, 0.0, 0.0]

        # Rendering variables
        self.viewer = None
        self.agent_transforms = None
        self.obstacle_transforms = None
        # self.resource_transforms = None

        try:
            self.agent_transforms = [rendering.Transform() for _ in range(self.num_agents)]
            self.obstacle_transforms = [rendering.Transform() for _ in range(self.num_obstacles_per_hall*3)]
            # self.resource_transforms = [rendering.Transform() for i in range(self.default_num_resources)]

        except:
            pass

        self.agent_positions = self.generate_agent_positions()

        # Step variables
        self.behaviour_map = [self.up, self.down, self.right, self.left]
        self.action_name = ["UP", "DOWN",  "RIGHT", "LEFT"]

        # Observation space
        # Agent's x-coordinate and y-coordinate
        self.observation_space_size = 2

        # Action space
        # 0- Forward, 1- Backward, 2- Right, 3- Left
        self.action_space_size = 4

    def step(self, agent_actions):
        """
                Updates the environment according to agents' actions

                :param agent_actions A list of integers representing the action each agent is taking

                :return A 4-tuple containing: a list containing each agent's observation, the reward at this time step,
                a boolean indicating if the simulation is done, any additional information
        """

        # Returns an error if the number of actions is incorrect
        assert len(agent_actions) == self.num_agents, "Incorrect number of actions"

        # Returns an error if any action is invalid
        for action in agent_actions:
            assert action in range(self.action_space_size), "%r (%s) invalid" % (action, type(action))

        rewards = [0.0] * len(agent_actions)

        action_successful = [False] * self.num_agents
        up_count = 0
        down_count = 0

        for i in range(self.num_agents):
            action_successful[i] = self.behaviour_map[agent_actions[i]](i)

            if self.action_name[agent_actions[i]] == "UP":
                up_count += 1

            elif self.action_name[agent_actions[i]] == "DOWN":
                down_count += 1

        self.num_pairs = min(up_count, down_count)

        for i in range(self.num_agents):
            # If an agent did a generalist behaviour
            if self.action_name[agent_actions[i]] == "RIGHT" and self.get_area_from_position(self.agent_positions[i]) == "RIGHT_HALL" and action_successful[i]:
                rewards[i] = 1
                self.total_rewarded_actions += 1

            # If an agent did a specialist behaviour
            elif self.action_name[agent_actions[i]] == "UP" and self.get_area_from_position(self.agent_positions[i]) == "UP_HALL" and action_successful[i]:
                rewards[i] = (2 * self.num_pairs) / up_count

                if self.num_pairs > 0:
                    self.specialised_actions += 1

                self.total_rewarded_actions += 1

            elif self.action_name[agent_actions[i]] == "DOWN" and self.get_area_from_position(self.agent_positions[i]) == "DOWN_HALL" and action_successful[i]:
                rewards[i] = (2 * self.num_pairs) / down_count

                if self.num_pairs > 0:
                    self.specialised_actions += 1

                self.total_rewarded_actions += 1

        observations = self.get_agent_observations()

        return observations, rewards

    def reset(self):
        try:
            self.viewer.close()
        except:
            pass

        self.viewer = None
        self.specialised_actions = 0
        self.total_rewarded_actions = 0
        self.agent_positions = self.generate_agent_positions()
        self.num_pairs = None

        return self.get_agent_observations()

    # Actions
    def up(self, agent_id):
        """
        agent with id agent_id moves one step forward i.e. up in the y direction
        :param agent_id: Index of the agent in self.agent_positions
        :return:
        """
        if self.agent_positions[agent_id][0] < self.arena_constraints["x_min"] + self.start_zone_size:
            y_min = self.arena_constraints["y_min"]
            y_max = self.arena_constraints["y_max"]
        else:
            y_min = self.arena_constraints["y_min"] + self.hall_size
            y_max = self.arena_constraints["y_min"] + self.hall_size + self.start_zone_size

        new_position = (self.agent_positions[agent_id][0],
                        np.clip(self.agent_positions[agent_id][1] + 1, y_min,
                                y_max - 1))

        if new_position not in self.obstacle_coordinates:
            self.agent_positions[agent_id] = new_position
            return True

        return False

    def down(self, agent_id):
        """
        agent with id agent_id moves one step back i.e. down in the y direction
        :param agent_id: Index of the agent in self.agent_positions
        :return:
        """
        if self.agent_positions[agent_id][0] < self.arena_constraints["x_min"] + self.start_zone_size:
            y_min = self.arena_constraints["y_min"]
            y_max = self.arena_constraints["y_max"]
        else:
            y_min = self.arena_constraints["y_min"] + self.hall_size
            y_max = self.arena_constraints["y_min"] + self.hall_size + self.start_zone_size

        new_position = (self.agent_positions[agent_id][0],
                        np.clip(self.agent_positions[agent_id][1] - 1, self.arena_constraints["y_min"],
                                self.arena_constraints["y_max"] - 1))

        if new_position not in self.obstacle_coordinates:
            self.agent_positions[agent_id] = new_position
            return True

        return False

    def right(self, agent_id):
        """
        agent with id agent_id moves one step to the right
        :param agent_id: Index of the agent in self.agent_positions
        :return:
        """
        x_min = self.arena_constraints["x_min"]

        if self.agent_positions[agent_id][1] >= self.arena_constraints["y_min"] + self.hall_size + self.start_zone_size or \
                self.agent_positions[agent_id][1] < self.arena_constraints["y_min"] + self.hall_size:

            x_max = self.arena_constraints["x_min"] + self.start_zone_size

        else:
            x_max = self.arena_constraints["x_max"]

        new_position = (np.clip(self.agent_positions[agent_id][0] + 1, x_min, x_max-1),
                        self.agent_positions[agent_id][1])

        if new_position not in self.obstacle_coordinates:
            self.agent_positions[agent_id] = new_position
            return True

        return False

    def left(self, agent_id):
        """
        agent with id agent_id moves one step to the left
        :param agent_id: Index of the agent in self.agent_positions
        :return:
        """
        x_min = self.arena_constraints["x_min"]

        if self.agent_positions[agent_id][1] >= self.arena_constraints["y_min"] + self.hall_size + self.start_zone_size or \
                self.agent_positions[agent_id][1] < self.arena_constraints["y_min"] + self.hall_size:

            x_max = self.arena_constraints["x_min"] + self.start_zone_size

        else:
            x_max = self.arena_constraints["x_max"]

        new_position = (np.clip(self.agent_positions[agent_id][0] - 1, x_min, x_max - 1),
                        self.agent_positions[agent_id][1])

        if new_position not in self.obstacle_coordinates:
            self.agent_positions[agent_id] = new_position
            return True

        return False

    # Simple getters --------------------------------------------------------------------------------------------------
    def get_num_agents(self):
        """
        Returns number of agents
        :return: Integer representing number of agents
        """
        return self.num_agents

    def get_observation_size(self):
        return self.observation_space_size

    def get_action_size(self):
        return self.action_space_size

    # Helpers
    def generate_agent_positions(self):
        return [(0, 0)]  * self.num_agents

    def generate_obstacle_position(self, hall):
        """
        Generates and returns valid coordinates for a single obstacle
        :return: x and y coordinates of the obstacle
        """
        x = self.np_random.randint(low=self.obstacle_generation_constraints[hall]["x_min"], high=self.obstacle_generation_constraints[hall]["x_max"])
        y = self.np_random.randint(low=self.obstacle_generation_constraints[hall]["y_min"], high=self.obstacle_generation_constraints[hall]["y_max"])
        return x, y

    def get_agent_observations(self):
        observations = [[0, 0]] * self.num_agents

        for i in range(self.num_agents):
            observations[i][0] = self.agent_positions[i][1]
            observations[i][1] = self.agent_positions[i][0]

        return observations

    def get_area_from_position(self, position):
        """
        Given an x,y coordinate, returns the name of the zone
        @param position:
        @return:
        """
        x = position[0]
        y = position[1]

        if x < self.arena_constraints["x_min"] or x > self.arena_constraints["x_max"]:
            raise ValueError("x position is not valid")
        if y < self.arena_constraints["y_min"] or x > self.arena_constraints["y_max"]:
            raise ValueError("y position is not valid")

        if y >= self.arena_constraints["y_min"] + self.hall_size + self.start_zone_size:
            return "UP_HALL"
        elif self.arena_constraints["y_min"] <= y < self.arena_constraints["y_min"] + self.hall_size:
            return "DOWN_HALL"
        elif x >= self.arena_constraints["x_min"] + self.start_zone_size:
            return "RIGHT_HALL"
        else:
            return "START_ZONE"

    def calculate_specialisation(self):
        if self.total_rewarded_actions == 0:
            return 0

        return [self.specialised_actions / self.total_rewarded_actions]

    def reset_rng(self):
        self.np_random = np.random.RandomState(self.seed_value)

    # Rendering functions
    def draw_arena_segment(self, top, bottom, left, right, rgb_tuple):
        """
        Helper function that creates the geometry for a segment of the arena. Intended to be used by the viewer

        :param top:
        :param bottom:
        :param rgb_tuple:
        :return: A FilledPolygon object that can be added to the viewer using add_geom
        """

        l, r, t, b = (left + self.x_shift) * self.scale, \
                     (right + self.x_shift) * self.scale, \
                     (top + self.y_shift) * self.scale, \
                     (bottom + self.y_shift) * self.scale
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
        verticals_list = np.linspace(0,
                                     self.total_width * self.scale,
                                     self.total_width + 1)

        for tick in verticals_list:
            xs = np.array([tick for _ in range(self.total_height + 1)])
            ys = np.linspace(0,
                             self.total_height * self.scale,
                             self.total_height + 1)
            xys = list(zip(xs, ys))

            line = rendering.make_polyline(xys)
            grid_lines += [line]

        # Draw horizontal lines
        horizontals_list = np.linspace(0,
                                       self.total_height * self.scale,
                                       self.total_height + 1)

        for tick in horizontals_list:
            xs = np.linspace(0,
                             self.total_width * self.scale,
                             self.total_width + 1)
            ys = np.array([tick for _ in range(self.total_width + 1)])
            xys = list(zip(xs, ys))

            line = rendering.make_polyline(xys)
            grid_lines += [line]

        return grid_lines

    def render(self, mode='human'):
        """
        Renders the environment, placing all agents in appropriate positions
        :param mode:
        :return:
        """

        screen_width = self.total_width * self.scale
        screen_height = self.total_height * self.scale

        if self.viewer is None:
            self.viewer = rendering.Viewer(screen_width, screen_height)

            # Draw up hall
            up_hall = self.draw_arena_segment(self.arena_constraints["y_max"],
                                              self.arena_constraints["y_max"] - self.hall_size,
                                              self.arena_constraints["x_min"],
                                              self.arena_constraints["x_min"] + self.start_zone_size,
                                              self.hall_colour)
            self.viewer.add_geom(up_hall)

            # Draw down hall
            down_hall = self.draw_arena_segment(self.arena_constraints["y_max"] - self.hall_size - self.start_zone_size,
                                                self.arena_constraints["y_min"],
                                                self.arena_constraints["x_min"],
                                                self.arena_constraints["x_min"] + self.start_zone_size,
                                                self.hall_colour)
            self.viewer.add_geom(down_hall)

            # Draw right hall
            right_hall = self.draw_arena_segment(self.arena_constraints["y_min"] + self.hall_size + self.start_zone_size,
                                                 self.arena_constraints["y_min"] + self.hall_size,
                                                 self.arena_constraints["x_min"] + self.start_zone_size,
                                                 self.arena_constraints["x_max"],
                                                 self.hall_colour)
            self.viewer.add_geom(right_hall)

            # Draw start zone
            start_zone = self.draw_arena_segment(self.arena_constraints["y_min"] + self.hall_size + self.start_zone_size,
                                                 self.arena_constraints["y_min"] + self.hall_size,
                                                 self.arena_constraints["x_min"],
                                                 self.arena_constraints["x_min"] + self.start_zone_size,
                                                 self.start_zone_colour)
            self.viewer.add_geom(start_zone)

            # Draw grid
            grid_lines = self.draw_grid()
            for line in grid_lines:
                self.viewer.add_geom(line)

            # Draw agent(s)
            for i in range(self.num_agents):
                agent = rendering.make_circle(self.agent_width / 2 * self.scale)
                agent.set_color(self.agent_colour[0], self.agent_colour[1], self.agent_colour[2])
                agent.add_attr(
                    rendering.Transform(
                        translation=(
                            0,
                            0)))
                agent.add_attr(self.agent_transforms[i])
                self.viewer.add_geom(agent)

            # Draw obstacles
            for i in range(len(self.obstacle_coordinates)):
                obstacle = rendering.make_circle(self.obstacle_width / 2 * self.scale)
                obstacle.set_color(self.obstacle_colour[0], self.obstacle_colour[1], self.obstacle_colour[2])
                obstacle.add_attr(
                    rendering.Transform(
                        translation=(
                            0,
                            0)))
                obstacle.add_attr(self.obstacle_transforms[i])
                self.viewer.add_geom(obstacle)

        # Set position of agent(s)
        for i in range(self.num_agents):
            self.agent_transforms[i].set_translation(
                (self.agent_positions[i][0] - self.arena_constraints["x_min"] + 0.5) * self.scale,
                (self.agent_positions[i][1] - self.arena_constraints["y_min"] + 0.5) * self.scale)

        # Set position of obstacle(s)
        for i, key in enumerate(self.obstacle_coordinates):
            self.obstacle_transforms[i].set_translation(
                (key[0] - self.arena_constraints["x_min"] + 0.5) * self.scale,
                (key[1] - self.arena_constraints["y_min"] + 0.5) * self.scale)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None