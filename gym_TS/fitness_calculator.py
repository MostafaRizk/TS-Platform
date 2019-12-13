import gym
import numpy as np

from agents.TinyAgent import TinyAgent
from gym.utils import seeding


class FitnessCalculator:
    def __init__(self, random_seed, simulation_length, num_trials, num_robots, num_resources, sensor_range, slope_angle, arena_length, arena_width, cache_start, slope_start, source_start):
        self.env = gym.make('gym_TS:TS-v1', num_robots=num_robots, num_resources=num_resources,
                            sensor_range=sensor_range, slope_angle=slope_angle, arena_length=arena_length, arena_width=arena_width,
                            cache_start=cache_start, slope_start=slope_start, source_start=source_start)
        # env = gym.wrappers.Monitor(env, 'video', force = True) # Uncomment to save video

        # Get size of input and output space and creates agent
        self.observation_size = self.env.get_observation_size()
        self.action_size = self.env.get_action_size()

        # Seeding values
        self.random_seed = random_seed
        self.np_random, self.random_seed = seeding.np_random(self.random_seed)

        # Action space uses a separate random number generator so need to set its seed separately
        self.env.action_space.np_random.seed(self.random_seed)

        self.simulation_length = simulation_length

        self.num_trials = num_trials

    def get_observation_size(self):
        return self.observation_size

    def get_action_size(self):
        return self.action_size

    def get_rng(self):
        return self.np_random

    def calculate_fitness(self, individual, learning_method="cma", render=False):
        """
        Calculates fitness of a controller by running a simulation
        :param render:
        :param num_trials:
        :param individual:
        :param learning_method Accepts cma or qn or bq
        :return:
        """

        #render = True
        average_score = 0
        temp_seed = self.random_seed

        for trial in range(self.num_trials):
            if learning_method == "cma" and not isinstance(individual, TinyAgent):
                temp_individual = TinyAgent(self.observation_size, self.action_size, temp_seed)
                temp_individual.load_weights(individual)
                individual = temp_individual

            self.env.seed(temp_seed)  # makes fitness deterministic
            observations = self.env.reset()

            if learning_method == "qn" or learning_method == "bq":
                for i in range(len(observations)):
                    observations[i] = np.array(observations[i])
                    observations[i] = np.reshape(observations[i], [1, self.get_observation_size()])

            score = 0
            done = False

            for t in range(self.simulation_length):
                if render:
                    self.env.render()

                # All agents act using same controller.
                robot_actions = [individual.act(observations[i]) for i in range(len(observations))]

                # The environment changes according to all their actions
                old_observations = observations[:]
                observations, reward, done, info = self.env.step(robot_actions, t)

                #if reward == 1:
                #    print(f"Agent got a reward at timestep {t}")

                if learning_method == "qn" or learning_method == "bq":
                    for i in range(len(observations)):
                        observations[i] = np.array(observations[i])
                        observations[i] = np.reshape(observations[i], [1, self.get_observation_size()])

                score += reward

                if learning_method == "qn" or learning_method == "bq":
                    for i in range(len(robot_actions)):
                        individual.remember(old_observations[i], robot_actions[i], reward, observations[i], done)

                #time.sleep(1)
                # print(f'Time: {t} || Score: {score}')

                if done:
                    break

            average_score += score
            temp_seed += 1

            if learning_method == "qn" or learning_method == "bq":
                loss = individual.replay()

        if learning_method == "qn" or learning_method == "bq":
            return average_score/self.num_trials, individual

        return average_score/self.num_trials

    def calculate_fitness_negation(self, individual, render=False):
        return -1*self.calculate_fitness(individual=individual, render=render)
