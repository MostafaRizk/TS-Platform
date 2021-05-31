import json
import numpy as np
import copy

# Put this import in a try/except because otherwise, the cluster throws an error
try:
    from helpers import rendering

except:
    pass


class ForkEnv:
    def __init__(self, parameter_filename=None):

        # Environment dimensions
        self.up_hall_size = 9
        self.down_hall_size = 9
        self.right_hall_size = 9
        self.start_zone_size = 3

        if self.start_zone_size % 2 == 0:
            self.offset = 0
        else:
            self.offset = 1

        self.arena_constraints = {"x_min": -(self.start_zone_size//2),
                                  "x_max": (self.start_zone_size//2) + self.offset + self.right_hall_size,
                                  "y_min": -(self.start_zone_size//2) - self.down_hall_size,
                                  "y_max": (self.start_zone_size//2) + self.offset + self.up_hall_size}

        self.total_width = self.start_zone_size + self.right_hall_size
        self.total_height = self.start_zone_size + self.up_hall_size + self.down_hall_size

        self.x_shift = abs(self.arena_constraints["x_min"])
        self.y_shift = abs(self.arena_constraints["y_min"])

        # Agent constants
        self.agent_width = 0.8

        # Other constants and variables
        self.num_agents = 2
        self.episode_length = 10

        # Rendering constants
        self.scale = 40
        self.start_zone_colour = [0.5, 0.5, 0.5]
        self.hall_colour = [0.25, 0.5, 0.5]
        self.agent_colour = [0, 0, 0.25]

        # Rendering variables
        self.viewer = None
        self.agent_transforms = None
        # self.resource_transforms = None

        try:
            self.agent_transforms = [rendering.Transform() for _ in range(self.num_agents)]
            # self.resource_transforms = [rendering.Transform() for i in range(self.default_num_resources)]

        except:
            pass

        self.agent_positions = [(-1, 0), (0, -1)]  # [None] * self.num_agents

        # Step variables
        self.behaviour_map = [self.up, self.down, self.right]
        self.action_name = ["UP", "DOWN",  "RIGHT"]

        self.seed_value = 1 #parameter_dictionary['general']['seed']
        self.np_random = np.random.RandomState(self.seed_value)

        # Observation space
        # Agent's x-coordinate and y-coordinate
        self.observation_space_size = 2

        # Action space
        # 0- Forward, 1- Backward, 2- Right
        self.action_space_size = 3

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

        teammate = [1, 0]

        for i in range(self.num_agents):
            self.behaviour_map[agent_actions[i]](i)

            # If an agent did a generalist behaviour
            if self.action_name[agent_actions[i]] == "RIGHT":
                rewards[i] = 1

            # If an agent did a specialist behaviour
            elif self.action_name[agent_actions[i]] == "UP" and self.action_name[agent_actions[teammate[i]]] == "DOWN":
                rewards[i] = 2
                rewards[teammate[i]] = 2

        observations = [[0, 0]] * self.num_agents

        for i in range(self.num_agents):
            observations[i][0] = self.agent_positions[i][1]
            observations[i][1] = self.agent_positions[i][0]

        return observations, rewards

    def reset(self):
        pass

    # Actions
    def up(self, agent_id):
        """
        agent with id agent_id moves one step forward i.e. up in the y direction
        :param agent_id: Index of the agent in self.agent_positions
        :return:
        """
        self.agent_positions[agent_id] = (
            self.agent_positions[agent_id][0],
            np.clip(self.agent_positions[agent_id][1] + 1, self.arena_constraints["y_min"],
                    self.arena_constraints["y_max"] - 1))

    def down(self, agent_id):
        """
        agent with id agent_id moves one step back i.e. down in the y direction
        :param agent_id: Index of the agent in self.agent_positions
        :return:
        """
        self.agent_positions[agent_id] = (
            self.agent_positions[agent_id][0],
            np.clip(self.agent_positions[agent_id][1] - 1, self.arena_constraints["y_min"],
                    self.arena_constraints["y_max"] - 1))

    def right(self, agent_id):
        """
        agent with id agent_id moves one step to the right
        :param agent_id: Index of the agent in self.agent_positions
        :return:
        """
        self.agent_positions[agent_id] = (
            np.clip(self.agent_positions[agent_id][0] + 1, self.arena_constraints["x_min"],
                    self.arena_constraints["x_max"] - 1),
            self.agent_positions[agent_id][1])

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
                                              self.arena_constraints["y_max"] - self.up_hall_size,
                                              self.arena_constraints["x_min"],
                                              self.arena_constraints["x_min"] + self.start_zone_size,
                                              self.hall_colour)
            self.viewer.add_geom(up_hall)

            # Draw down hall
            down_hall = self.draw_arena_segment(self.arena_constraints["y_max"] - self.up_hall_size - self.start_zone_size,
                                                self.arena_constraints["y_min"],
                                                self.arena_constraints["x_min"],
                                                self.arena_constraints["x_min"] + self.start_zone_size,
                                                self.hall_colour)
            self.viewer.add_geom(down_hall)

            # Draw right hall
            right_hall = self.draw_arena_segment(self.arena_constraints["y_min"] + self.down_hall_size + self.start_zone_size,
                                                 self.arena_constraints["y_min"] + self.down_hall_size,
                                                 self.arena_constraints["x_min"] + self.start_zone_size,
                                                 self.arena_constraints["x_max"],
                                                 self.hall_colour)
            self.viewer.add_geom(right_hall)

            # Draw start zone
            start_zone = self.draw_arena_segment(self.arena_constraints["y_min"] + self.down_hall_size + self.start_zone_size,
                                                 self.arena_constraints["y_min"] + self.down_hall_size,
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

        # Set position of agent(s)
        for i in range(self.num_agents):
            self.agent_transforms[i].set_translation(
                (self.agent_positions[i][0] - self.arena_constraints["x_min"] + 0.5) * self.scale,
                (self.agent_positions[i][1] - self.arena_constraints["y_min"] + 0.5) * self.scale)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None