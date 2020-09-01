import json
import unittest

from envs.slope import SlopeEnv


class SlopeEnvTest(unittest.TestCase):
    def create_parameter_file(self):
        filename = "test_parameters.json"
        parameter_dictionary = {
            "general": {
                "algorithm_selected": "cma",
                "team_type": "heterogeneous",
                "reward_level": "team",
                "agent_type": "nn",
                "environment": "slope",
                "seed": 1
            },
            "environment": {
                "slope": {
                    "num_agents": 2,
                    "num_resources": 4,
                    "sensor_range": 1,
                    "sliding_speed": 4,
                    "arena_length": 8,
                    "arena_width": 4,
                    "cache_start": 1,
                    "slope_start": 3,
                    "source_start": 7,
                    "base_cost": 1,
                    "upward_cost_factor": 3.0,
                    "downward_cost_factor": 0.2,
                    "carry_factor": 2,
                    "resource_reward": 1000,
                    "simulation_length": 500,
                    "num_simulation_runs": 5
                }
            },
            "agent": {
                "nn": {
                    "architecture": "rnn",
                    "bias": "False",
                    "hidden_layers": 0,
                    "hidden_units_per_layer": 4,
                    "activation_function": "linear"
                }
            },
            "algorithm": {
                "agent_population_size": 10,
                "rwg": {
                    "sampling_distribution": "normal",
                    "normal": {
                        "mean": 0,
                        "std": 1
                    },
                    "uniform": {
                        "min": -3,
                        "max": 3
                    }
                },
                "cma": {
                    "sigma": 0.2,
                    "generations": 5,
                    "tolx": 1e-3,
                    "tolfunhist": 2e2
                }
            }
        }
        f = open(filename, "w")
        dictionary_string = json.dumps(parameter_dictionary, indent=4)
        f.write(dictionary_string)
        f.close()
        return filename

    def edit_parameter_file(self, parameter_filename, new_dictionary):
        f = open(parameter_filename, "w")
        dictionary_string = json.dumps(new_dictionary, indent=4)
        f.write(dictionary_string)
        f.close()

    def test_step(self):
        # Test: When an agent does a pickup action one step away from a resource, it picks it up
        parameter_filename = self.create_parameter_file()
        parameter_dictionary = json.loads(open(parameter_filename).read())
        parameter_dictionary['environment']['slope']['num_agents'] = 1
        self.edit_parameter_file(parameter_filename, parameter_dictionary)
        env = SlopeEnv(parameter_filename)
        env.reset()
        for i in range(int(parameter_dictionary['environment']['slope']['source_start'])):
            env.step([0])
        env.step([4])
        self.assertTrue(env.has_resource[0] is not None)

    def test_act_and_reward(self):
        # Test: When sliding speed is 0, reward for moving up vs down slope is the same
        parameter_filename = self.create_parameter_file()
        parameter_dictionary = json.loads(open(parameter_filename).read())
        parameter_dictionary['environment']['slope']['sliding_speed'] = 0
        self.edit_parameter_file(parameter_filename, parameter_dictionary)
        env = SlopeEnv(parameter_filename)
        env.reset()
        env.step([0, 0])
        env.step([0, 0])
        env.step([0, 0])
        env.step([0, 0])
        rewards_up = env.act_and_reward([0, 0])
        rewards_down = env.act_and_reward([1, 1])
        self.assertEqual(rewards_up, rewards_down)

        # Test: When sliding speed is 1, reward for moving up the slope is -4 and down is -0.8
        parameter_filename = self.create_parameter_file()
        parameter_dictionary = json.loads(open(parameter_filename).read())
        parameter_dictionary['environment']['slope']['sliding_speed'] = 1
        self.edit_parameter_file(parameter_filename, parameter_dictionary)
        env = SlopeEnv(parameter_filename)
        env.reset()
        env.step([0, 0])
        env.step([0, 0])
        env.step([0, 0])
        env.step([0, 0])
        rewards_up = env.act_and_reward([0, 0])
        self.assertTrue(rewards_up == [-4.0, -4.0])
        rewards_down = env.act_and_reward([1, 1])
        self.assertTrue(rewards_down == [-0.8, -0.8])

        # Test: When a resource is retrieved, the reward is the same whether there are 2 agents or 1
        # 1 agent
        parameter_filename = self.create_parameter_file()
        parameter_dictionary = json.loads(open(parameter_filename).read())
        parameter_dictionary['environment']['slope']['num_agents'] = 1
        self.edit_parameter_file(parameter_filename, parameter_dictionary)
        env = SlopeEnv(parameter_filename)
        env.reset()
        for i in range(int(parameter_dictionary['environment']['slope']['source_start']) + 1):
            env.step([0])
        env.step([4])
        for i in range(int(parameter_dictionary['environment']['slope']['source_start']) + 1):
            env.step([1])
        self.assertTrue(env.has_resource[0] is not None)
        observation, reward = env.step([5])
        estimated_reward = int(parameter_dictionary['environment']['slope']['resource_reward'])
        self.assertTrue(reward[0] == estimated_reward-1)
        # 2 agents
        parameter_filename = self.create_parameter_file()
        parameter_dictionary = json.loads(open(parameter_filename).read())
        env = SlopeEnv(parameter_filename)
        env.reset()
        for i in range(int(parameter_dictionary['environment']['slope']['source_start']) + 1):
            env.step([0, 0])
        env.step([5, 4])
        for i in range(int(parameter_dictionary['environment']['slope']['source_start']) + 1):
            env.step([1, 1])
        self.assertTrue(env.has_resource[0] is None)
        self.assertTrue(env.has_resource[1] is not None)
        observation, reward = env.step([5, 5])
        estimated_reward = int(parameter_dictionary['environment']['slope']['resource_reward'])
        self.assertTrue(reward[0]+reward[1] == estimated_reward-2)


    if __name__ == '__main__':
        unittest.main()
