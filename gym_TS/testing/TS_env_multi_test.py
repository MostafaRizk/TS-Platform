import unittest
import gym
from gym import spaces
import numpy as np


class TestMultiEnv(unittest.TestCase):
    def create_testing_env(self, width=8, length=12):
        env = gym.make('gym_TS:TS-v1')
        env.seed(1)

        # Set environment size and layout
        env.arena_constraints = {"x_min": 0, "x_max": width, "y_min": 0, "y_max": length}
        env.nest_size = env.arena_constraints["y_max"] / 12
        env.cache_size = env.nest_size * 2
        env.slope_size = env.nest_size * 6
        env.source_size = env.nest_size * 3
        env.nest_start = env.arena_constraints["y_min"]
        env.cache_start = env.nest_start + env.nest_size
        env.slope_start = env.cache_start + env.cache_size
        env.source_start = env.slope_start + env.slope_size

        return env

    def test_generate_arena(self):
        env = gym.make('gym_TS:TS-v1')

        env.arena_constraints = {"x_min": 0, "x_max": 8, "y_min": 0, "y_max": 12}
        self.assertEqual(env.generate_arena(), [[0, 0, 0, 0, 0, 0, 0, 0],
                                                [0, 0, 0, 0, 0, 0, 0, 0],
                                                [0, 0, 0, 0, 0, 0, 0, 0],
                                                [0, 0, 0, 0, 0, 0, 0, 0],
                                                [0, 0, 0, 0, 0, 0, 0, 0],
                                                [0, 0, 0, 0, 0, 0, 0, 0],
                                                [0, 0, 0, 0, 0, 0, 0, 0],
                                                [0, 0, 0, 0, 0, 0, 0, 0],
                                                [0, 0, 0, 0, 0, 0, 0, 0],
                                                [0, 0, 0, 0, 0, 0, 0, 0],
                                                [0, 0, 0, 0, 0, 0, 0, 0],
                                                [0, 0, 0, 0, 0, 0, 0, 0]])

        env.arena_constraints = {"x_min": 0, "x_max": 2, "y_min": 0, "y_max": 4}
        self.assertEqual(env.generate_arena(), [[0, 0],
                                                [0, 0],
                                                [0, 0],
                                                [0, 0]])

    def test_generate_robot_position(self):
        env = gym.make('gym_TS:TS-v1')
        env.seed(1)

        self.assertTrue(isinstance(env.generate_robot_position(), tuple))

        env.arena_constraints = {"x_min": 0, "x_max": 8, "y_min": 0, "y_max": 12}
        env.nest_size = env.arena_constraints["y_max"] / 12
        env.cache_size = env.nest_size * 2
        env.slope_size = env.nest_size * 6
        env.source_size = env.nest_size * 3
        env.nest_start = env.arena_constraints["y_min"]
        env.cache_start = env.nest_start + env.nest_size
        env.slope_start = env.cache_start + env.cache_size
        env.source_start = env.slope_start + env.slope_size

        for i in range(100):
            x, y = env.generate_robot_position()
            self.assertTrue(0 <= x < 8)
            self.assertTrue(0 <= y < 1)

    def test_generate_robot_observations(self):
        # Robot is in the corner of the nest with sensing range 0 and has no object
        env = self.create_testing_env()
        env.num_robots = 1
        env.sensor_range = 0
        env.tiles_in_sensing_range = (2 * env.sensor_range + 1) ** 2
        env.observation_space = spaces.Discrete(env.tiles_in_sensing_range * 4 + 5)
        env.reset()
        env.robot_map = env.generate_arena()
        env.resource_map = env.generate_arena()
        env.robot_positions = [(0, 0)]
        env.robot_map[0][0] = 1
        env.has_resource = [None]
        self.assertTrue(np.array_equal(env.generate_robot_observations(), [np.array([1, 0, 0, 0, 1, 0, 0, 0, 0])]))

        # Robot is in the corner of the nest with sensing range 0 and has an object
        env = self.create_testing_env()
        env.num_robots = 1
        env.default_num_resources = 1
        env.current_num_resources = env.default_num_resources
        env.latest_resource_id = env.default_num_resources - 1
        env.sensor_range = 0
        env.tiles_in_sensing_range = (2 * env.sensor_range + 1) ** 2
        env.observation_space = spaces.Discrete(env.tiles_in_sensing_range * 4 + 5)
        env.reset()
        env.robot_map = env.generate_arena()
        env.resource_map = env.generate_arena()
        env.robot_positions = [(0, 0)]
        env.resource_positions = [(0, 0)]
        env.robot_map[0][0] = 1
        env.resource_map[0][0] = 1
        env.has_resource = [0]
        obs = env.generate_robot_observations()
        self.assertTrue(np.array_equal(env.generate_robot_observations(), [np.array([1, 0, 0, 0, 1, 0, 0, 0, 1])]))

        # Robot is in the top center of the source with sensing range 0 and has an object
        env = self.create_testing_env()
        env.num_robots = 1
        env.default_num_resources = 1
        env.current_num_resources = env.default_num_resources
        env.latest_resource_id = env.default_num_resources - 1
        env.sensor_range = 0
        env.tiles_in_sensing_range = (2 * env.sensor_range + 1) ** 2
        env.observation_space = spaces.Discrete(env.tiles_in_sensing_range * 4 + 5)
        env.reset()
        env.robot_map = env.generate_arena()
        env.resource_map = env.generate_arena()
        env.robot_positions = [(4, 11)]
        env.resource_positions = [(4, 11)]
        env.robot_map[11][4] = 1
        env.resource_map[11][4] = 1
        env.has_resource = [0]
        self.assertTrue(np.array_equal(env.generate_robot_observations(), [np.array([1, 0, 0, 0, 0, 0, 0, 1, 1])]))

        # Robot is in the middle of the slope with sensing range 0 and has no object but one is present
        env = self.create_testing_env()
        env.num_robots = 1
        env.default_num_resources = 1
        env.current_num_resources = env.default_num_resources
        env.latest_resource_id = env.default_num_resources - 1
        env.sensor_range = 0
        env.tiles_in_sensing_range = (2 * env.sensor_range + 1) ** 2
        env.observation_space = spaces.Discrete(env.tiles_in_sensing_range * 4 + 5)
        env.reset()
        env.robot_map = env.generate_arena()
        env.resource_map = env.generate_arena()
        env.robot_positions = [(4, 7)]
        env.resource_positions = [(4, 7)]
        env.robot_map[7][4] = 1
        env.resource_map[7][4] = 1
        env.has_resource = [None]
        self.assertTrue(np.array_equal(env.generate_robot_observations(), [np.array([0, 0, 1, 0, 0, 0, 1, 0, 0])]))

        # Two robots, near each other, sensing range 0, resource adjacent to one of them
        env = self.create_testing_env()
        env.num_robots = 2
        env.default_num_resources = 1
        env.current_num_resources = env.default_num_resources
        env.latest_resource_id = env.default_num_resources - 1
        env.sensor_range = 0
        env.tiles_in_sensing_range = (2 * env.sensor_range + 1) ** 2
        env.observation_space = spaces.Discrete(env.tiles_in_sensing_range * 4 + 5)
        env.robot_positions = [None for i in range(env.num_robots)]
        env.resource_positions = [None for i in range(env.default_num_resources)]
        env.reset()
        env.robot_map = env.generate_arena()
        env.resource_map = env.generate_arena()
        env.robot_positions = [(0, 0), (4, 0)]
        env.resource_positions = [(3, 0)]
        env.robot_map[0][0] = 1
        env.robot_map[4][0] = 2
        env.resource_map[0][3] = 1
        env.has_resource = [None, None]
        self.assertTrue(np.array_equal(env.generate_robot_observations(), [np.array([1, 0, 0, 0, 1, 0, 0, 0, 0]),
                                                                           np.array([1, 0, 0, 0, 1, 0, 0, 0, 0])]))

        # Two robots, near each other, sensing range 1, resource adjacent to one of them
        env = self.create_testing_env()
        env.num_robots = 2
        env.default_num_resources = 1
        env.current_num_resources = env.default_num_resources
        env.latest_resource_id = env.default_num_resources - 1
        env.sensor_range = 1
        env.tiles_in_sensing_range = (2 * env.sensor_range + 1) ** 2
        env.observation_space = spaces.Discrete(env.tiles_in_sensing_range * 4 + 5)
        env.robot_positions = [None for i in range(env.num_robots)]
        env.resource_positions = [None for i in range(env.default_num_resources)]
        env.reset()
        env.robot_map = env.generate_arena()
        env.resource_map = env.generate_arena()
        env.robot_positions = [(0, 0), (4, 0)]
        env.resource_positions = [(3, 0)]
        env.robot_map[0][0] = 1
        env.robot_map[0][4] = 2
        env.resource_map[0][3] = 1
        env.has_resource = [None, None]
        self.assertTrue(np.array_equal(env.generate_robot_observations(), [np.array([0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0,
                                                                                     0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0,
                                                                                     0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1,
                                                                                     1, 0, 0, 0, 0]),
                                                                           np.array([1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0,
                                                                                     0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0,
                                                                                     0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1,
                                                                                     1, 0, 0, 0, 0])]))

        # Two robots, near each other, sensing range 1, resource adjacent to one of them
        env = self.create_testing_env()
        env.num_robots = 2
        env.default_num_resources = 1
        env.current_num_resources = env.default_num_resources
        env.latest_resource_id = env.default_num_resources - 1
        env.sensor_range = 1
        env.tiles_in_sensing_range = (2 * env.sensor_range + 1) ** 2
        env.observation_space = spaces.Discrete(env.tiles_in_sensing_range * 4 + 5)
        env.robot_positions = [None for i in range(env.num_robots)]
        env.resource_positions = [None for i in range(env.default_num_resources)]
        env.reset()
        env.robot_map = env.generate_arena()
        env.resource_map = env.generate_arena()
        env.robot_positions = [(0, 0), (2, 0)]
        env.resource_positions = [(1, 0)]
        env.robot_map[0][0] = 1
        env.robot_map[0][2] = 2
        env.resource_map[0][1] = 1
        env.has_resource = [None, None]
        obs = env.generate_robot_observations()
        self.assertTrue(np.array_equal(env.generate_robot_observations(), [np.array([0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0,
                                                                                     0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0,
                                                                                     0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1,
                                                                                     1, 0, 0, 0, 0]),
                                                                           np.array([1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0,
                                                                                     0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0,
                                                                                     0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1,
                                                                                     1, 0, 0, 0, 0])]))

        # Two robots, near each other, sensing range 2, resource adjacent to one of them, both in range of each other
        env = self.create_testing_env()
        env.num_robots = 2
        env.default_num_resources = 1
        env.current_num_resources = env.default_num_resources
        env.latest_resource_id = env.default_num_resources - 1
        env.sensor_range = 2
        env.tiles_in_sensing_range = (2 * env.sensor_range + 1) ** 2
        env.observation_space = spaces.Discrete(env.tiles_in_sensing_range * 4 + 5)
        env.robot_positions = [None for i in range(env.num_robots)]
        env.resource_positions = [None for i in range(env.default_num_resources)]
        env.reset()
        env.robot_map = env.generate_arena()
        env.resource_map = env.generate_arena()
        env.robot_positions = [(0, 0), (2, 0)]
        env.resource_positions = [(1, 0)]
        env.robot_map[0][0] = 1
        env.robot_map[0][2] = 2
        env.resource_map[0][1] = 1
        env.has_resource = [None, None]
        obs = env.generate_robot_observations()
        self.assertTrue(np.array_equal(env.generate_robot_observations(),
                                       [np.array([0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0,
                                                  0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0,
                                                  0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0,
                                                  0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1,
                                                  0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1,
                                                  1, 0, 0, 0, 0]),
                                        np.array([1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0,
                                                  1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0,
                                                  0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0,
                                                  0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1,
                                                  0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1,
                                                  1, 0, 0, 0, 0])]))

        # Two robots, near each other, sensing range 1, resource adjacent to both of them, both in range, one has resource, one is on top of a resource it doesn't currently have
        env = self.create_testing_env()
        env.num_robots = 2
        env.default_num_resources = 3
        env.current_num_resources = env.default_num_resources
        env.latest_resource_id = env.default_num_resources - 1
        env.sensor_range = 1
        env.tiles_in_sensing_range = (2 * env.sensor_range + 1) ** 2
        env.observation_space = spaces.Discrete(env.tiles_in_sensing_range * 4 + 5)
        env.robot_positions = [None for i in range(env.num_robots)]
        env.resource_positions = [None for i in range(env.default_num_resources)]
        env.reset()
        env.robot_map = env.generate_arena()
        env.resource_map = env.generate_arena()
        env.robot_positions = [(1, 1), (2, 1)]  # Both of the robots are in the cache
        env.resource_positions = [(1, 1), (2, 1), (2, 2)]
        env.robot_map[1][1] = 1
        env.robot_map[1][2] = 2
        env.resource_map[1][1] = 1
        env.resource_map[1][2] = 2
        env.resource_map[2][2] = 3
        env.has_resource = [None, 1]
        obs = env.generate_robot_observations()
        self.assertTrue(np.array_equal(env.generate_robot_observations(), [np.array([1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0,
                                                                                     1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0,
                                                                                     1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0,
                                                                                     0, 1, 0, 0, 0]),
                                                                           np.array([1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0,
                                                                                     0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0,
                                                                                     1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0,
                                                                                     0, 1, 0, 0, 1])]))

    def test_step(self):
        # Check that a robot carrying a resource cannot walk into another resource
        env = self.create_testing_env()
        env.num_robots = 1
        env.default_num_resources = 2
        env.current_num_resources = env.default_num_resources
        env.latest_resource_id = env.default_num_resources - 1
        env.sensor_range = 1
        env.tiles_in_sensing_range = (2 * env.sensor_range + 1) ** 2
        env.observation_space = spaces.Discrete(env.tiles_in_sensing_range * 4 + 5)
        env.robot_positions = [None for i in range(env.num_robots)]
        env.resource_positions = [None for i in range(env.default_num_resources)]
        env.reset()
        env.robot_map = env.generate_arena()
        env.resource_map = env.generate_arena()
        env.robot_positions = [(2, 10)]
        env.resource_positions = [(2, 10), (2, 11)]
        env.robot_map[10][2] = 1
        env.resource_map[10][2] = 1
        env.resource_map[11][2] = 2
        env.has_resource = [0]

        actions = [0]  # Move forward
        time_step = 0
        env.step(actions, time_step)
        self.assertEqual(env.robot_map[10][2], 1)
        self.assertEqual(env.robot_positions[0], (2, 10))

        # Check that robot can move backward
        actions = [1]
        env.step(actions, time_step)
        self.assertEqual(env.robot_positions[0], (2, 9))
        self.assertEqual(env.robot_map[10][2], 0)
        self.assertEqual(env.robot_map[9][2], 1)

        # Check that robot can move left
        actions = [2]
        env.step(actions, time_step)
        self.assertEqual(env.robot_positions[0], (1, 9))
        self.assertEqual(env.robot_map[9][2], 0)
        self.assertEqual(env.robot_map[9][1], 1)

        # Check that robot can move forward
        actions = [0]
        env.step(actions, time_step)
        self.assertEqual(env.robot_positions[0], (1, 10))
        self.assertEqual(env.robot_map[9][1], 0)
        self.assertEqual(env.robot_map[10][1], 1)

        # Check that robot can move right
        actions = [3]
        env.step(actions, time_step)
        self.assertEqual(env.robot_positions[0], (2, 10))
        self.assertEqual(env.robot_map[10][1], 0)
        self.assertEqual(env.robot_map[10][2], 1)

        # Check that robot can drop
        actions = [5]
        env.step(actions, time_step)
        # Check resource is where it should be
        self.assertEqual(env.resource_map[10][2], 1)
        self.assertEqual(env.resource_positions[0], (2, 10))
        # Check robot doesn't have it
        self.assertEqual(env.has_resource[0], None)
        # Check robot is still where it should be
        self.assertEqual(env.robot_positions[0], (2, 10))
        self.assertEqual(env.robot_map[10][2], 1)

        # Check that robot can pick up
        actions = [3]  # Right
        env.step(actions, time_step)
        self.assertEqual(env.resource_map[10][2], 1)
        self.assertEqual(env.resource_positions[0], (2, 10))
        actions = [0]  # Forward
        env.step(actions, time_step)
        actions = [2]  # Left
        env.step(actions, time_step)
        actions = [4]  # Pick up
        env.step(actions, time_step)
        # Check resource is where it should be
        self.assertEqual(env.resource_map[11][2], 2)
        self.assertEqual(env.resource_positions[1], (2, 11))
        # Check robot has it
        self.assertEqual(env.has_resource[0], 1)

        # Check that robot can move while carrying resource

    if __name__ == '__main__':
        unittest.main()
