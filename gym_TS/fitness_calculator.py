import gym
import numpy as np
import time

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

    def calculate_fitness(self, individual1, team_type, learning_method="cma", render=False):
        """
        Calculates fitness of a controller by running a simulation
        :param individual1:
        :param team_type Accepts "homogeneous" or "heterogeneous"
        :param learning_method Accepts cma. Also accepts qn or bq but will only work for homogeneous teams
        :param render:
        :return:
        """

        #render = True
        average_score = 0
        temp_seed = self.random_seed
        full_genome = None

        if not isinstance(individual1, TinyAgent):
            full_genome = individual1
        else:
            full_genome = individual1.get_weights()

        individual2 = None

        for trial in range(self.num_trials):
            if team_type == "homogeneous":
                if learning_method == "cma":
                    temp_individual = TinyAgent(self.observation_size, self.action_size, temp_seed)
                    temp_individual.load_weights(full_genome)
                    individual1 = temp_individual
            elif team_type == "heterogeneous":
                mid = int(len(full_genome) / 2)
                temp_individual = TinyAgent(self.observation_size, self.action_size, temp_seed)
                temp_individual.load_weights(full_genome[0:mid])
                individual1 = temp_individual
                temp_individual.load_weights(full_genome[mid:])
                individual2 = temp_individual

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

                robot_actions = []

                if team_type == "homogeneous":
                    # All agents act using same controller.
                    robot_actions = [individual1.act(observations[i]) for i in range(len(observations))]
                    #robot_actions = [self.env.action_space.sample() for i in range(len(observations))]  # Random actions for testing
                elif team_type == "heterogeneous":
                    for i in range(len(observations)):
                        if i % 2 == 0:
                            robot_actions += [individual1.act(observations[i])]
                        else:
                            robot_actions += [individual2.act(observations[i])]

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
                        individual1.remember(old_observations[i], robot_actions[i], reward, observations[i], done)

                #time.sleep(0.1)
                #print(f'Time: {t} || Score: {score}')

                if done:
                    break

            average_score += score
            temp_seed += 1

            if learning_method == "qn" or learning_method == "bq":
                loss = individual1.replay()

        if learning_method == "qn" or learning_method == "bq":
            return average_score/self.num_trials, individual1

        return average_score/self.num_trials

    def calculate_ferrante_specialisation(self, individual1, team_type, learning_method="cma", render=False):
        """
        Calculates fitness of a controller by running a simulation
        :param individual1:
        :param team_type Accepts "homogeneous" or "heterogeneous"
        :param learning_method Accepts cma. Also accepts qn or bq but will only work for homogeneous teams
        :param render:
        :return:
        """

        #render = True
        average_score = 0
        average_specialisation = 0
        temp_seed = self.random_seed
        full_genome = None

        if not isinstance(individual1, TinyAgent):
            full_genome = individual1
        else:
            full_genome = individual1.get_weights()

        individual2 = None

        for trial in range(self.num_trials):
            if team_type == "homogeneous":
                if learning_method == "cma":
                    temp_individual = TinyAgent(self.observation_size, self.action_size, temp_seed)
                    temp_individual.load_weights(full_genome)
                    individual1 = temp_individual
            elif team_type == "heterogeneous":
                mid = int(len(full_genome) / 2)
                temp_individual = TinyAgent(self.observation_size, self.action_size, temp_seed)
                temp_individual.load_weights(full_genome[0:mid])
                individual1 = temp_individual
                temp_individual.load_weights(full_genome[mid:])
                individual2 = temp_individual

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

                robot_actions = []

                if team_type == "homogeneous":
                    # All agents act using same controller.
                    robot_actions = [individual1.act(observations[i]) for i in range(len(observations))]
                    #robot_actions = [self.env.action_space.sample() for i in range(len(observations))]  # Random actions for testing
                elif team_type == "heterogeneous":
                    for i in range(len(observations)):
                        if i % 2 == 0:
                            robot_actions += [individual1.act(observations[i])]
                        else:
                            robot_actions += [individual2.act(observations[i])]

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
                        individual1.remember(old_observations[i], robot_actions[i], reward, observations[i], done)

                #time.sleep(1)
                #print(f'Time: {t} || Score: {score}')

                if done:
                    break

            average_score += score
            average_specialisation += self.env.calculate_ferrante_specialisation()
            temp_seed += 1

            if learning_method == "qn" or learning_method == "bq":
                loss = individual1.replay()

        if learning_method == "qn" or learning_method == "bq":
            return average_score/self.num_trials, average_specialisation/self.num_trials, individual1

        return average_score/self.num_trials, average_specialisation/self.num_trials

    def calculate_fitness_negation(self, individual, team_type, render=False):
        #return -1*self.calculate_fitness(individual1=individual, team_type=team_type, render=True)#render)
        return -1 * self.calculate_fitness(individual1=individual, team_type=team_type, render=render)

