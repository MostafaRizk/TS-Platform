import json
import numpy as np
import copy

# Put this import in a try/except because otherwise, the cluster throws an error
try:
    from helpers import rendering

except:
    pass


class BoxPushingEnv:
    def __init__(self, parameter_filename=None):
        if parameter_filename is None:
            raise RuntimeError("No parameter file specified for the environment")

        parameter_dictionary = json.loads(open(parameter_filename).read())

        self.seed_value = parameter_dictionary['environment']['box_pushing']['env_seed']
        self.np_random = np.random.RandomState(self.seed_value)

        # Cooperation parameters

        if parameter_dictionary['environment']['box_pushing']['defection'] == "True":
            self.defection = True
        elif parameter_dictionary['environment']['box_pushing']['defection'] == "False":
            self.defection = False

        if parameter_dictionary['environment']['box_pushing']['partial_cooperation'] == "True":
            self.partial_cooperation = True
        elif parameter_dictionary['environment']['box_pushing']['partial_cooperation'] == "False":
            self.partial_cooperation = False

        self.sensing = parameter_dictionary['environment']['box_pushing']['sensing']

        if parameter_dictionary['environment']['box_pushing']['specialisation'] == "True":
            self.specialisation = True
        elif parameter_dictionary['environment']['box_pushing']['specialisation'] == "False":
            self.specialisation = False

        self.environment_scaling = parameter_dictionary['environment']['box_pushing']['environment_scaling']

        if parameter_dictionary['environment']['box_pushing']['reward_sharing'] == "True":
            self.reward_sharing = True
        elif parameter_dictionary['environment']['box_pushing']['reward_sharing'] == "False":
            self.reward_sharing = False

        self.time_scaling = parameter_dictionary['environment']['box_pushing']['time_scaling']

        if parameter_dictionary['environment']['box_pushing']['sparse_rewards'] == "True":
            self.sparse_rewards = True
        elif parameter_dictionary['environment']['box_pushing']['sparse_rewards'] == "False":
            self.sparse_rewards = False

        # Environment

        self.num_agents = parameter_dictionary['environment']['box_pushing']['num_agents']
        self.arena_length = parameter_dictionary['environment']['box_pushing']['arena_length']
        self.arena_width = parameter_dictionary['environment']['box_pushing']['min_arena_width'] * self.num_agents

        if self.defection:
            self.num_small_boxes = self.num_agents
        else:
            self.num_small_boxes = 0

        if self.partial_cooperation:
            self.medium_box_size = parameter_dictionary['environment']['box_pushing']['medium_box_size']
            self.num_medium_boxes = self.num_agents // self.medium_box_size
            self.num_large_boxes = 0
        else:
            self.num_medium_boxes = 0
            self.num_large_boxes = 1  # 1 box pushed by all agents

        if self.time_scaling == "variable":
            self.episode_length = parameter_dictionary['environment']['box_pushing']['min_episode_length'] * self.num_agents
        elif self.time_scaling == "constant":
            self.episode_length = parameter_dictionary['environment']['box_pushing']['min_episode_length']
        else:
            raise RuntimeError("Time scaling must be either constant or variable")

        self.num_episodes = parameter_dictionary['environment']['box_pushing']['num_episodes']

        self.arena_constraints = {"x_min": 0,
                                  "x_max": self.arena_width,
                                  "y_min": 0,
                                  "y_max": self.arena_length}

        self.home_length = self.goal_length = 1

        self.agent_width = 0.8
        self.small_box_width = 0.8

        if self.partial_cooperation:
            self.medium_box_width = self.medium_box_size - 0.2

        self.large_box_width = self.num_agents - 0.2
        self.agent_height = 0.4

        # Rewards
        self.large_reward_per_agent = parameter_dictionary['environment']['box_pushing']['large_reward_per_agent']
        self.small_reward_per_agent = parameter_dictionary['environment']['box_pushing']['small_reward_per_agent']
        self.cost_per_time_step = parameter_dictionary['environment']['box_pushing']['cost_per_time_step']

        # Rendering constants
        self.scale = 40
        self.goal_colour = [0.5, 0.5, 0.5]
        self.main_colour = [0.25, 0.5, 0.5]
        self.home_colour = [0.5, 0.6, 0.6]
        self.agent_colour = [0, 0, 0.25]
        self.box_colour = [0, 0.25, 0]

        # Rendering variables
        self.viewer = None
        self.agent_transforms = None
        self.box_transforms = None

        try:
            self.agent_transforms = [rendering.Transform() for _ in range(self.num_agents)]
            self.box_transforms = [rendering.Transform() for _ in range(self.num_large_boxes + self.num_medium_boxes + self.num_small_boxes)]

        except:
            pass

        self.agent_positions = [None] * self.num_agents
        self.boxes_in_arena = {}

        # Step variables
        self.behaviour_map = [self.forward, self.rotate_right, self.rotate_left, self.stay]
        self.action_name = ["FORWARD", "ROTATE RIGHT", "ROTATE LEFT", "STAY"]

        # Observation space
        if self.sensing == "local":
            # Onehotencoded vector, with possibilities for each of the 3 box sizes, an agent, a wall, or an empty spot
            self.observation_space_size = 6

        # Action space
        # 0- Forward, 1- Rotate right, 2- Rotate left, 3- Stay
        self.action_space_size = 4

        # Given an agent's orientation, add these values to the agent's x and y to get the block in front of them
        self.orientation_map = {"NORTH": (0, 1),
                                "SOUTH": (0, -1),
                                "EAST": (1, 0),
                                "WEST": (-1, 0)}

    def step(self, agent_actions):
        observations = [0] * self.observation_space_size
        rewards = 0
        return observations, rewards

    def reset(self):
        try:
            self.viewer.close()
        except:
            pass

        self.viewer = None

        # Creates empty state
        self.agent_map = self.generate_arena()  # Empty agent map
        self.box_map = self.generate_arena() # Empty box map
        self.boxes_in_arena = {}

        # Places all agents
        for i in range(self.num_agents):
            agent_placed = False
            while not agent_placed:
                x, y, orientation = self.generate_agent_position()
                if self.agent_map[y][x] == 0:
                    self.agent_map[y][x] = i + 1
                    self.agent_positions[i] = (x, y, orientation)
                    agent_placed = True

        # Places all boxes
        for i in range(self.num_small_boxes + self.num_medium_boxes + self.num_large_boxes):
            box_placed = False

            while not box_placed:
                x, y = self.generate_box_position()

                # Place small boxes
                if i < self.num_small_boxes:
                    #if self.box_map[y][x] == 0:
                    if all([self.box_map[j][x] == 0 for j in range(self.arena_constraints["y_max"])]):
                        self.box_map[y][x] = i + 1
                        self.boxes_in_arena[i] = (x, y, "small")
                        box_placed = True

                # Place medium boxes
                elif i >= self.num_small_boxes and self.num_medium_boxes > 0:
                    if (x + self.medium_box_size - 1) < self.arena_constraints["x_max"]:
                        if all([self.box_map[y][x+j] == 0 for j in range(self.medium_box_size)]) and \
                                all([all([self.box_map[k][x+j] == 0 for j in range(self.medium_box_size)]) for k in range(self.arena_constraints["y_max"])]):

                            for j in range(self.medium_box_size):
                                self.box_map[y][x+j] = i + 1

                            self.boxes_in_arena[i] = (x, y, "medium")
                            box_placed = True

                # Place large box
                elif i >= self.num_small_boxes and self.num_large_boxes > 0:
                    if (x + self.num_agents - 1) < self.arena_constraints["x_max"]:
                        if all([self.box_map[y][x+j] == 0 for j in range(self.num_agents)]) and \
                                all([all([self.box_map[k][x+j] == 0 for j in range(self.num_agents)]) for k in range(self.arena_constraints["y_max"])]):

                            for j in range(self.num_agents):
                                self.box_map[y][x+j] = i + 1

                            self.boxes_in_arena[i] = (x, y, "large")
                            box_placed = True

        return self.get_agent_observations()

    # Initialisers ----------------------------------------------------------------------------------------------------
    def generate_arena(self):
        """
        Generates 2D matrix representing an empty arena i.e. each tile contains a 0
        :return:
        """

        return [[0 for _ in range(self.arena_constraints["x_max"])] for _ in range(self.arena_constraints["y_max"])]

    def generate_agent_position(self):
        """
        Generates and returns valid coordinates for a single agent
        :return: x and y coordinates of the agent
        """
        x = self.np_random.randint(low=self.arena_constraints["x_min"], high=self.arena_constraints["x_max"])
        y = self.arena_constraints["y_min"]
        orientation = "NORTH"
        return x, y, orientation

    def generate_box_position(self):
        """
        Generates and returns valid coordinates for a single resource
        :return: x and y coordinates of the resource
        """
        x = self.np_random.randint(low=self.arena_constraints["x_min"], high=self.arena_constraints["x_max"])
        y = self.np_random.randint(low=self.home_length, high=self.arena_constraints["y_max"]-self.goal_length)
        return x, y

    def get_agent_observations(self):
        """
        Generate a list containing each agent's observation. Depending on the sensing parameter, each agent observes:

        Sensing = local
        The contents of the tile in front of it (Empty, Wall, Agent, Small box, Medium box, Large box)

        Sensing = partial (Needs to be implemented)
        The locations of all resources and its own location

        Sensing = full (Needs to be implemented)
        The whole state space

        :return:
        """
        observations = []
        readable_observations = []

        for j in range(len(self.agent_positions)):
            position = self.agent_positions[j]
            agent_x = position[0]
            agent_y = position[1]
            agent_orientation = position[2]
            agent_observation = [0]*self.get_observation_size()

            # x and y of whatever is in front of the agent
            y_in_front = agent_y + self.orientation_map[agent_orientation][1]
            x_in_front = agent_x + self.orientation_map[agent_orientation][0]

            if y_in_front < self.arena_constraints["y_min"] or y_in_front >= self.arena_constraints["y_max"] or \
                    x_in_front < self.arena_constraints["x_min"] or x_in_front >= self.arena_constraints["x_max"]:
                agent_observation[1] = 1
                readable_agent_observation = "Wall"

            # if there is a box in front of the agent
            elif self.box_map[y_in_front][x_in_front] != 0:
                box_id = self.box_map[y_in_front][x_in_front]
                box_size = self.boxes_in_arena[box_id-1][2]

                if box_size == "small":
                    agent_observation[3] = 1
                    readable_agent_observation = "Small box"

                elif box_size == "medium":
                    agent_observation[4] = 1
                    readable_agent_observation = "Medium box"

                elif box_size == "large":
                    agent_observation[5] = 1
                    readable_agent_observation = "Large box"

                else:
                    raise RuntimeError("Box size not supported")

            # If there is an agent in front
            elif self.agent_map[y_in_front][x_in_front] != 0:
                agent_observation[2] = 1
                readable_agent_observation = "Agent"

            # If there is nothing in front
            else:
                agent_observation[0] = 1
                readable_agent_observation = "Empty"

            observations += [np.array(agent_observation)]
            readable_observations += [readable_agent_observation]

        return observations

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

    def get_episode_length(self):
        return self.episode_length

    # Helpers
    def reset_rng(self):
        self.np_random = np.random.RandomState(self.seed_value)

    # Actions
    def forward(self):
        pass

    def rotate_right(self):
        pass

    def rotate_left(self):
        pass

    def stay(self):
        pass

    # Rendering functions
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
                                     self.arena_constraints["x_max"] * self.scale,
                                     self.arena_constraints["x_max"] + 1)

        for tick in verticals_list:
            xs = np.array([tick for _ in range(self.arena_constraints["y_max"] + 1)])
            ys = np.linspace(self.arena_constraints["y_min"] * self.scale,
                             self.arena_constraints["y_max"] * self.scale,
                             self.arena_constraints["y_max"] + 1)
            xys = list(zip(xs, ys))

            line = rendering.make_polyline(xys)
            grid_lines += [line]

        # Draw horizontal lines
        horizontals_list = np.linspace(self.arena_constraints["y_min"] * self.scale,
                                       self.arena_constraints["y_max"] * self.scale,
                                       self.arena_constraints["y_max"] + 1)

        for tick in horizontals_list:
            xs = np.linspace(self.arena_constraints["x_min"] * self.scale,
                             self.arena_constraints["x_max"] * self.scale,
                             self.arena_constraints["x_max"] + 1)
            ys = np.array([tick for _ in range(self.arena_constraints["x_max"] + 1)])
            xys = list(zip(xs, ys))

            line = rendering.make_polyline(xys)
            grid_lines += [line]

        return grid_lines

    def render(self, mode='human'):
        """
        Renders the environment, placing all agents and boxes in appropriate positions
        :param mode:
        :return:
        """

        screen_width = self.arena_constraints["x_max"] * self.scale
        screen_height = self.arena_constraints["y_max"] * self.scale

        if self.viewer is None:
            self.viewer = rendering.Viewer(screen_width, screen_height)
            home_top = self.arena_constraints["y_min"] + self.home_length
            main_top = self.arena_constraints["y_max"] - self.goal_length
            goal_top = self.arena_constraints["y_max"]

            # Draw home
            home = self.draw_arena_segment(home_top, self.arena_constraints["y_min"], self.home_colour)
            self.viewer.add_geom(home)

            # Draw main
            main_area = self.draw_arena_segment(main_top, home_top, self.main_colour)
            self.viewer.add_geom(main_area)

            # Draw goal
            goal = self.draw_arena_segment(goal_top, main_top, self.goal_colour)
            self.viewer.add_geom(goal)

            # Draw grid
            grid_lines = self.draw_grid()
            for line in grid_lines:
                self.viewer.add_geom(line)

            # Draw agent(s)
            for i in range(self.num_agents):
                agent = rendering.FilledPolygon([(0, 0), (self.agent_width*self.scale, 0), (self.agent_width/2*self.scale, self.agent_height*self.scale)])
                agent.set_color(self.agent_colour[0], self.agent_colour[1], self.agent_colour[2])
                agent.add_attr(
                    rendering.Transform(
                        translation=(
                            -self.agent_width/2*self.scale,
                            -self.agent_height/2*self.scale)))
                agent.add_attr(self.agent_transforms[i])
                self.viewer.add_geom(agent)

            # Draw box(es)
            for i in range(self.num_small_boxes + self.num_medium_boxes + self.num_large_boxes):
                if i < self.num_small_boxes and self.num_small_boxes > 0:
                    l, r, t, b = -self.small_box_width / 2 * self.scale, self.small_box_width / 2 * self.scale, self.small_box_width * self.scale, 0

                elif i >= self.num_small_boxes and self.num_medium_boxes > 0:
                    l, r, t, b = -self.medium_box_width / 2 * self.scale, self.medium_box_width / 2 * self.scale, self.small_box_width * self.scale, 0

                elif i >= self.num_small_boxes and self.num_large_boxes > 0:
                    l, r, t, b = -self.large_box_width / 2 * self.scale, self.large_box_width / 2 * self.scale, self.small_box_width * self.scale, 0

                box = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
                box.set_color(self.box_colour[0], self.box_colour[1], self.box_colour[2])
                box.add_attr(
                    rendering.Transform(
                        translation=(
                            0,
                            0)))
                box.add_attr(self.box_transforms[i])
                self.viewer.add_geom(box)

        # Set position of agent(s)
        for i in range(self.num_agents):
            self.agent_transforms[i].set_translation(
                (self.agent_positions[i][0] - self.arena_constraints["x_min"] + 0.5) * self.scale,
                (self.agent_positions[i][1] - self.arena_constraints["y_min"] + 0.5) * self.scale)

        # Set position of box(es)
        for box_id, box_details in self.boxes_in_arena.items():
            if box_details[2] == "small":
                self.box_transforms[box_id].set_translation(
                    (box_details[0] - self.arena_constraints["x_min"] + 0.5) * self.scale,
                    (box_details[1] - self.arena_constraints["y_min"] + 0.125) * self.scale)

            elif box_details[2] == "medium":
                self.box_transforms[box_id].set_translation(
                    (box_details[0] - self.arena_constraints["x_min"] + (0.5 * self.medium_box_size)) * self.scale,
                    (box_details[1] - self.arena_constraints["y_min"] + 0.125) * self.scale)

            elif box_details[2] == "large":
                self.box_transforms[box_id].set_translation(
                    (box_details[0] - self.arena_constraints["x_min"] + (0.5 * self.num_agents)) * self.scale,
                    (box_details[1] - self.arena_constraints["y_min"] + 0.125) * self.scale)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
