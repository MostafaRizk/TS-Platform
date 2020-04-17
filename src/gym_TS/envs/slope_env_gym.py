from src.gym_TS.envs.slope_env_parent import SlopeEnvParent

import numpy as np
import copy

import gym
from gym import spaces
from gym.utils import seeding

from gym.envs.classic_control import rendering


class SlopeEnvGym(SlopeEnvParent, gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self, num_robots=4, num_resources=5, sensor_range=1, slope_angle=20, arena_length=12, arena_width=8,
                 cache_start=1, slope_start=3, source_start=9, upward_cost_factor=3, downward_cost_factor=0.2,
                 carry_factor=1, resource_reward_factor=1000):
        SlopeEnvParent.__init__(self, num_robots=4, num_resources=5, sensor_range=1, slope_angle=20, arena_length=12, arena_width=8,
                 cache_start=1, slope_start=3, source_start=9, upward_cost_factor=3, downward_cost_factor=0.2,
                 carry_factor=1, resource_reward_factor=1000)

        try:
            self.robot_transforms = [rendering.Transform() for i in range(self.num_robots)]
            self.resource_transforms = [rendering.Transform() for i in range(self.default_num_resources)]

        except:
            pass

        # Observation space
        #
        # The details are explained in self.generate_robot_observations()
        self.tiles_in_sensing_range = (2 * self.sensor_range + 1) ** 2  # Range=1 -> 9 tiles. Range=2 -> 25 tiles. Robot at the center.
        self.observation_space = spaces.Discrete(self.tiles_in_sensing_range * 4 + 4 + 1)  # Tiles are onehotencoded. 4 bits for possible locations, 1 bit for object possession

        # Action space
        self.action_space = spaces.Discrete(6)  # 0- Forward, 1- Backward, 2- Left, 3- Right, 4- Pick up, 5- Drop

    def seed(self, seed=None):
        """
                Generates random seed for np.random
                :param seed:
                :return:
                """
        self.np_random, seed = seeding.np_random(seed)
        self.seed_value = seed
        return [seed]

    def step(self, robot_actions, time_step):
        """
        Given the current state of the environment and the actions the robots are taking, updates the environment accordingly

        :param robot_actions A list of integers representing the action each robot is taking
        :param time_step The current time step within the simulation

        :return A 4-tuple containing: a list containing each robot's observation, the reward at this time step,
        a boolean indicating if the simulation is done, any additional information

        IMPORTANT: Function assumes that, if controllers are different, odd numbered robots use one controller type and
        even numbered controllers use another controller type
        """

        # Returns an error if the number of actions is incorrect
        assert len(robot_actions) == self.num_robots, "Incorrect number of actions"

        # Returns an error if any action is invalid
        for action in robot_actions:
            assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        done = False

        rewards = [0.0, 0.0]

        # The robots act
        old_robot_positions = copy.deepcopy(self.robot_positions)

        for i in range(len(robot_actions)):
            cost_multiplier = 1
            team_id = i % 2

            # If robot is carrying something, multiply the cost of moving
            if self.has_resource[i] is not None:
                cost_multiplier = self.carry_factor

            if robot_actions[i] < 4:
                self.behaviour_map[robot_actions[i]](i)

                # More costly for robot to move up the slope than down
                if self.get_area_from_position(self.robot_positions[i]) == "SLOPE":
                    if self.action_name[robot_actions[i]] == "FORWARD":
                        rewards[team_id] -= self.upward_cost_factor *self.base_cost*cost_multiplier

                    elif self.action_name[robot_actions[i]] == "BACKWARD":
                        rewards[team_id] -= self.base_cost*self.downward_cost_factor*cost_multiplier

                    else:
                        rewards[team_id] -= self.base_cost*cost_multiplier

                # Negative reward for moving when not on slope. Same as having a battery
                else:
                    rewards[team_id] -= self.base_cost*cost_multiplier

            # Negative reward for dropping/picking up but is not affected by resource weight
            else:
                rewards[team_id] -= self.base_cost

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
                    if self.has_resource[i] is not None and self.has_resource[i] != j:
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

        # If a robot has returned a resource to the nest the resource is deleted and the robot is rewarded
        for i in range(self.num_robots):
            if self.get_area_from_position(self.robot_positions[i]) == "NEST" and self.has_resource[i] is None and self.robot_positions[i] in self.resource_positions:
                #self.has_resource[i] = None
                resource_id = self.resource_positions.index(self.robot_positions[i]) # Find the resource with the same position as the current robot and get that resource's id

                # Reward all robots if a resource is retrieved
                rewards[0] += self.reward_for_resource
                rewards[1] += self.reward_for_resource

                self.delete_resource(resource_id)
                #self.spawn_resource()

        num_resources_at_source = 0

        # Spawn a new resource any time the number of resources at the source decreases below the default threshold
        for position in self.resource_positions:
            try:
                if self.get_area_from_position(position) == "SOURCE":
                    num_resources_at_source += 1
            except ValueError:
                pass

            # If there are more resources at the source than the default, there's no need to continue counting
            if num_resources_at_source >= self.default_num_resources:
                break

        resource_deficit = self.default_num_resources - num_resources_at_source
        #print(num_resources_at_source)

        if resource_deficit > 0:
            for i in range(resource_deficit):
                self.spawn_resource()

        # Update the state with the new resource positions
        for i in range(len(self.resource_positions)):
            if self.resource_positions[i] != self.dumping_position:
                #self.state[self.resource_positions[i][1] + self.arena_constraints["y_max"]][self.resource_positions[i][0]] = i + 1
                self.resource_map[self.resource_positions[i][1]][self.resource_positions[i][0]] = i + 1

        self.state = np.concatenate((self.robot_map, self.resource_map), axis=0)  # Fully observable environment
        observations = self.generate_robot_observations()

        #return self.state, reward, done, {}
        return observations, sum(rewards), done, {"reward_1": rewards[0], "reward_2": rewards[1]}

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
        assert self.num_robots <= self.arena_constraints[
            "x_max"] * self.nest_size, "Not enough room in the nest for all robots"
        assert self.default_num_resources <= self.arena_constraints[
            "x_max"] * self.source_size, "Not enough room in the source for all resources"

        try:
            self.viewer.close()
        except:
            pass

        self.viewer = None

        self.resource_positions = [None for i in range(self.default_num_resources)]

        self.resource_carried_by = [[] for i in range(self.default_num_resources)]

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
            if self.resource_map[y][x] == 0 and (x, y) not in self.resource_positions:
                self.resource_map[y][x] = self.latest_resource_id + 1
                self.latest_resource_id += 1
                self.resource_positions += [(x, y)]
                # self.resource_positions[self.latest_resource_id] = (x, y)
                try:
                    self.resource_transforms += [rendering.Transform()]
                except:
                    pass
                self.resource_carried_by += [[]]
                resource_placed = True
                self.current_num_resources += 1
                try:
                    self.add_resource_to_rendering(self.latest_resource_id)
                except:
                    pass
                return x, y

