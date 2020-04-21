import gym
import numpy as np
import time

from gym_TS.agents.TinyAgent import TinyAgent
from gym.utils import seeding
from gym_TS.envs.slope_env_gymless import SlopeEnvGymless

from gym_TS.agents.HardcodedCollectorAgent import HardcodedCollectorAgent
from gym_TS.agents.HardcodedDropperAgent import HardcodedDropperAgent
from gym_TS.agents.HardcodedGeneralistAgent import HardcodedGeneralistAgent


class FitnessCalculator:

    def __init__(self, random_seed, simulation_length, num_trials, num_robots, num_resources, sensor_range, slope_angle,
                 arena_length, arena_width, cache_start, slope_start, source_start, upward_cost_factor, downward_cost_factor,
                 carry_factor, resource_reward_factor, using_gym=False):

        if using_gym:
            self.env = gym.make('gym_package.gym_TS:TS-v1', num_robots=num_robots, num_resources=num_resources,
                                sensor_range=sensor_range, slope_angle=slope_angle, arena_length=arena_length,
                                arena_width=arena_width, cache_start=cache_start, slope_start=slope_start,
                                source_start=source_start, upward_cost_factor=upward_cost_factor,
                                downward_cost_factor=downward_cost_factor, carry_factor=carry_factor,
                                resource_reward_factor=resource_reward_factor)
            # env = gym.wrappers.Monitor(env, 'video', force = True) # Uncomment to save video

        else:
            self.env = SlopeEnvGymless(num_robots=num_robots, num_resources=num_resources,
                                sensor_range=sensor_range, slope_angle=slope_angle, arena_length=arena_length,
                                arena_width=arena_width, cache_start=cache_start, slope_start=slope_start,
                                source_start=source_start, upward_cost_factor=upward_cost_factor,
                                downward_cost_factor=downward_cost_factor, carry_factor=carry_factor,
                                resource_reward_factor=resource_reward_factor)

        # Get size of input and output space and creates agent
        self.observation_size = self.env.get_observation_size()
        self.action_size = self.env.get_action_size()

        # Seeding values
        self.random_seed = random_seed
        self.np_random, self.random_seed = seeding.np_random(self.random_seed)

        if using_gym:
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

    def calculate_fitness_of_population(self, population, team_type, selection_level, render=False):
        """
        Calculates fitness of entire population

        :param population:
        :param team_type:
        :param selection_level:
        :param learning_method:
        :param render:
        :return: List of fitnesses, one for each member of the population
        """
        fitnesses = []

        if team_type == "homogeneous" and selection_level == "team":
            for genome in population:
                individual_1 = genome
                individual_2 = genome
                fitness_1, fitness_2 = self.calculate_fitness(team_type, selection_level, individual_1, individual_2, render)
                team_fitness = fitness_1 + fitness_2
                fitnesses += [team_fitness]

        elif team_type == "heterogeneous" and selection_level == "team":
            for genome in population:
                mid = int(len(genome) / 2)
                individual_1 = genome[0:mid]
                individual_2 = genome[mid:]
                fitness_1, fitness_2 = self.calculate_fitness(team_type, selection_level, individual_1,
                                                              individual_2, render)
                team_fitness = fitness_1 + fitness_2
                fitnesses += [team_fitness]

        elif (team_type == "heterogeneous" and selection_level == "individual") or \
            (team_type == "homogeneous" and selection_level == "individual"):
            for i in range(0, len(population), 2):
                individual_1 = population[i]
                individual_2 = population[i+1]
                fitness_1, fitness_2 = self.calculate_fitness(team_type, selection_level, individual_1,
                                                              individual_2, render)

                fitnesses += [fitness_1, fitness_2]

        else:
            raise RuntimeError("Invalid team type and/or selection level")

        return fitnesses

    def calculate_fitness(self, team_type, selection_level, individual_1, individual_2, render=False):
        """
        Calculates fitness of a controller by running a simulation
        :param individual_1: Genome (NN weights)
        :param individual_2: Genome (NN weights)
        :param team_type Accepts "homogeneous" or "heterogeneous"
        :param selection_level Accepts "individual" or "team"
        :param learning_method Accepts cma. Also accepts qn or bq but will only work for homogeneous teams
        :param render:
        :return:
        """

        #render = True
        average_score = 0
        temp_seed = self.random_seed

        # Load genomes into TinyAgent objects (i.e. neural networks)
        temp_individual_1 = TinyAgent(self.observation_size, self.action_size, temp_seed)
        temp_individual_2 = TinyAgent(self.observation_size, self.action_size, temp_seed)
        temp_individual_1.load_weights(individual_1)
        temp_individual_2.load_weights(individual_2)
        individual_1 = temp_individual_1
        individual_2 = temp_individual_2

        # For use with individual level selection
        average_score_1 = 0
        average_score_2 = 0

        for trial in range(self.num_trials):
            self.env.seed(temp_seed)  # makes fitness deterministic
            observations = self.env.reset()

            score = 0

            # For use with individual selection
            score_1 = 0
            score_2 = 0

            for t in range(self.simulation_length):
                if render:
                    self.env.render()

                robot_actions = []

                for i in range(len(observations)):
                    if i % 2 == 0:
                        robot_actions += [individual_1.act(observations[i])]
                    else:
                        robot_actions += [individual_2.act(observations[i])]

                # The environment changes according to all their actions
                observations, reward, done, info = self.env.step(robot_actions, t)

                # Team selection
                score += reward

                # Individual selection
                score_1 += info["reward_1"]
                score_2 += info["reward_2"]

                #time.sleep(0.1)
                #print(f'Time: {t} || Score: {score}')

                if done:
                    break

            # Team selection
            average_score += score

            #Individual selection
            average_score_1 += score_1
            average_score_2 += score_2

            temp_seed += 1

        return average_score_1/self.num_trials, average_score_2/self.num_trials

    def calculate_ferrante_specialisation(self, individual_1, team_type, learning_method="cma", render=False):
        """
        Calculates fitness of a controller by running a simulation
        :param individual_1:
        :param team_type Accepts "homogeneous" or "heterogeneous"
        :param learning_method Accepts cma. Also accepts qn or bq but will only work for homogeneous teams
        :param render:
        :return:
        """

        #render = True
        average_score = 0
        average_specialisation = 0
        temp_seed = self.random_seed
        full_genome_1 = None

        if not isinstance(individual_1, TinyAgent):
            full_genome_1 = individual_1
        else:
            full_genome_1 = individual_1.get_weights()

        individual_2 = None

        for trial in range(self.num_trials):
            if team_type == "homogeneous":
                if learning_method == "cma":
                    temp_individual = TinyAgent(self.observation_size, self.action_size, temp_seed)
                    temp_individual.load_weights(full_genome_1)
                    individual_1 = temp_individual
            elif team_type == "heterogeneous":
                mid = int(len(full_genome_1) / 2)
                temp_individual = TinyAgent(self.observation_size, self.action_size, temp_seed)
                temp_individual.load_weights(full_genome_1[0:mid])
                individual_1 = temp_individual
                temp_individual.load_weights(full_genome_1[mid:])
                individual_2 = temp_individual

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
                    robot_actions = [individual_1.act(observations[i]) for i in range(len(observations))]
                    #robot_actions = [self.env.action_space.sample() for i in range(len(observations))]  # Random actions for testing
                elif team_type == "heterogeneous":
                    for i in range(len(observations)):
                        if i % 2 == 0:
                            robot_actions += [individual_1.act(observations[i])]
                        else:
                            robot_actions += [individual_2.act(observations[i])]

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
                        individual_1.remember(old_observations[i], robot_actions[i], reward, observations[i], done)

                #time.sleep(0.1)
                #print(f'Time: {t} || Score: {score}')

                if done:
                    break

            average_score += score
            average_specialisation += self.env.calculate_ferrante_specialisation()
            temp_seed += 1

            if learning_method == "qn" or learning_method == "bq":
                loss = individual_1.replay()

        if learning_method == "qn" or learning_method == "bq":
            return average_score/self.num_trials, average_specialisation/self.num_trials, individual_1

        return average_score/self.num_trials, average_specialisation/self.num_trials

    def calculate_fitness_negation(self, individual, team_type, render=False):
        #return -1*self.calculate_fitness(individual_1=individual, team_type=team_type, render=True)#render)
        return -1 * self.calculate_fitness(individual_1=individual, team_type=team_type, render=render)

    def calculate_hardcoded_fitness(self, type, render=False):
        """
        Calculates fitness of a controller by running a simulation
        :param individual_1:
        :param team_type Accepts "homogeneous" or "heterogeneous"
        :param learning_method Accepts cma. Also accepts qn or bq but will only work for homogeneous teams
        :param render:
        :return:
        """

        #render = True
        average_score = 0
        average_specialisation = 0
        temp_seed = self.random_seed
        individual_1 = None
        individual_2 = None

        if type == "generalist":
            individual_1 = HardcodedGeneralistAgent()
            individual_2 = HardcodedGeneralistAgent()
        elif type == "specialist":
            individual_1 = HardcodedDropperAgent()
            individual_2 = HardcodedCollectorAgent()
        elif type == "mixed_dropper":
            individual_1 = HardcodedGeneralistAgent()
            individual_2 = HardcodedDropperAgent()
        elif type == "mixed_collector":
            individual_1 = HardcodedGeneralistAgent()
            individual_2 = HardcodedCollectorAgent()
        elif type == "double_dropper":
            individual_1 = HardcodedDropperAgent()
            individual_2 = HardcodedDropperAgent()
        elif type == "double_collector":
            individual_1 = HardcodedCollectorAgent()
            individual_2 = HardcodedCollectorAgent()
        else:
            raise RuntimeError("Hardcoding type must be either generalist or specialist")

        for trial in range(self.num_trials):
            self.env.seed(temp_seed)  # makes fitness deterministic
            observations = self.env.reset()

            score = 0
            done = False

            for t in range(self.simulation_length):
                if render:
                    self.env.render()

                robot_actions = []

                for i in range(len(observations)):
                    if i % 2 == 0:
                        robot_actions += [individual_1.act(observations[i])]
                    else:
                        robot_actions += [individual_2.act(observations[i])]

                # The environment changes according to all their actions
                old_observations = observations[:]
                observations, reward, done, info = self.env.step(robot_actions, t)

                score += reward

                #time.sleep(0.1)
                #print(f'Time: {t} || Score: {score}')

                if done:
                    break

            average_score += score
            average_specialisation += self.env.calculate_ferrante_specialisation()
            temp_seed += 1

        return average_score/self.num_trials, average_specialisation/self.num_trials

