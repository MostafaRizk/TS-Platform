import unittest
import gym

class TestMultiEnv(unittest.TestCase):
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

    if __name__ == '__main__':
        unittest.main()