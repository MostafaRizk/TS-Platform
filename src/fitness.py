import copy
import json
import os
import numpy as np
import time
from envs.slope import SlopeEnv


class FitnessCalculator:

    def __init__(self, parameter_filename):

        if parameter_filename is None:
            raise RuntimeError("No parameter file specified for the fitness function")

        self.parameter_filename = parameter_filename
        self.parameter_dictionary = json.loads(open(parameter_filename).read())

        if self.parameter_dictionary["general"]["environment"] == "slope":
            self.env = SlopeEnv(parameter_filename)

        environment_name = self.parameter_dictionary['general']['environment']
        self.num_agents = self.parameter_dictionary['environment'][environment_name]['num_agents']

        # Get size of input and output space (for use in agent creation)
        self.observation_size = self.env.get_observation_size()
        self.action_size = self.env.get_action_size()

        # Seeding values
        self.random_seed = self.parameter_dictionary['general']['seed']
        self.np_random = np.random.RandomState(self.random_seed)

        self.episode_length = self.parameter_dictionary['environment']['slope']['episode_length']

        self.num_episodes = self.parameter_dictionary['environment']['slope']['num_episodes']

    def calculate_fitness_of_agent_population(self, population):
        """
        Takes a population of Agent objects, places each group in a team and calculates the fitnesses of each

        @param population: List of Agent objects
        @return: List containing fitness value of each Agent in the population
        """

        assert len(population) % self.num_agents == 0, "Population needs to be divisible by the number of agents per team"

        fitnesses = []
        agents_per_team = self.num_agents

        # Get fitnesses of each team of agents
        for i in range(0, len(population), agents_per_team):
            agent_list = [population[i+j] for j in range(0, agents_per_team)]
            fitness_dict = self.calculate_fitness(agent_list)
            fitness_matrix = fitness_dict['fitness_matrix']

            fitnesses += fitness_matrix

        return fitnesses

    def calculate_fitness(self, agent_list, render=False, time_delay=0, measure_specialisation=False,
                          logging=False, logfilename=None, render_mode="human"):
        """
        Calculates the fitness of a team of agents. Fitness is calculated
        by running the simulation for t time steps (as specified in the parameter file) with each agent acting every
        time step based on its observations. The simulation is restarted at the end and repeated a certain number of
        times (according to the parameter file). This is done without resetting the random number generator so that
        there are new initial positions for agents and resources in each simulation run.

        @param agent_list: List of Agent objects
        @param render: Boolean indicating whether or not simulations will be visualised
        @param time_delay: Integer indicating how many seconds delay (for smoother visualisation)
        @param measure_specialisation: Boolean indicating whether or not specialisation is being measured
        @param logging: Boolean indicating whether or not actions will be logged
        @param logfilename: String name of the file where actions will be logged
        @param render_mode: If rgb_array, creates an rgb array of the first episode. Otherwise plays video. Only does either if render is True
        @return: Dictionary containing 'fitness_matrix' (each index i is a list of agent i's fitnesses for every episode)
        and 'specialisation_list' (a measure of the degree of
        specialisation observed for the team for each episode)
        """

        assert len(agent_list) == self.num_agents, "Agents passed to function do not match parameter file"

        # Initialise major variables
        file_reader = None
        fitness_matrix = [[0]*self.num_episodes for i in range(len(agent_list))]
        specialisation_list = []
        agent_copies = [copy.deepcopy(agent) for agent in agent_list]
        video_frames = []

        # Create logging file if logging
        if logging:
            if not logfilename:
                raise RuntimeError("Cannot log results without naming the logfile")

            if not os.path.exists(logfilename):
                file_reader = open(logfilename, "w+")
                header_line = ','.join(f"t={t}" for t in range(self.episode_length)) + "\n"
                file_reader.write(header_line)
            else:
                file_reader = open(logfilename, "a")

        # Run the simulation several times
        for episode in range(self.num_episodes):
            observations = self.env.reset()

            # Initialise variables
            agent_action_matrix = [[-1]*self.episode_length for i in range(len(agent_list))]
            current_epsiode_reward_matrix = [[0]*self.episode_length for i in range(len(agent_list))]

            # Do 1 run of the simulation
            for t in range(self.episode_length):
                if render and render_mode == "rgb_array" and episode == 0:
                    video_frames += [self.env.render(mode=render_mode)]
                elif render:
                    self.env.render(mode=render_mode)

                robot_actions = []

                for i in range(len(observations)):
                    robot_actions += [agent_copies[i].act(observations[i])]
                    agent_action_matrix[i][t] = [robot_actions[-1]]

                # The environment changes according to all their actions
                observations, rewards = self.env.step(robot_actions)

                # Calculate how much of the rewards go to each agent type
                for i in range(len(rewards)):
                    fitness_matrix[i][episode] += rewards[i]
                    current_epsiode_reward_matrix[i][t] = rewards[i]

                if time_delay > 0:
                    time.sleep(time_delay)

            # Reset agent networks
            agent_copies = [copy.deepcopy(agent) for agent in agent_list]

            # Extra computations if calculating specialisation or logging actions
            if measure_specialisation:
                specialisation_list += [self.env.calculate_ferrante_specialisation()]

            if logging:
                #for agent_action_list in agent_action_matrix:
                #    action_string = ','.join(str(action) for action in agent_action_list) + '\n'
                #    file_reader.write(action_string)
                for reward_list in current_epsiode_reward_matrix:
                    reward_string = ','.join(str(reward) for reward in reward_list) + '\n'
                    file_reader.write(reward_string)

        if logging:
            file_reader.close()

        return {"fitness_matrix": fitness_matrix, "specialisation_list": specialisation_list, "video_frames": video_frames}

    # Helpers ---------------------------------------------------------------------------------------------------------

    def get_observation_size(self):
        return self.observation_size

    def get_action_size(self):
        return self.action_size

    def get_rng(self):
        return self.np_random

    def get_parameter_filename(self):
        return self.parameter_filename

    def get_parameter_dictionary(self):
        return self.parameter_dictionary

