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

        # Get size of input and output space (for use in agent creation)
        self.observation_size = self.env.get_observation_size()
        self.action_size = self.env.get_action_size()

        # Seeding values
        self.random_seed = self.parameter_dictionary['general']['seed']
        self.np_random = np.random.RandomState(self.random_seed)

        self.simulation_length = self.parameter_dictionary['environment']['simulation_length']

        self.num_simulation_runs = self.parameter_dictionary['environment']['num_simulation_runs']

    def calculate_fitness_of_agent_population(self, population):
        """
        Takes a population of Agent objects, places each pair on a team and calculates the fitnesses of each

        @param population: List of Agent objects
        @return: List containing fitness value of each Agent in the population
        """

        fitnesses = []

        for i in range(0, len(population), 2):
            agent_1 = population[i]
            agent_2 = population[i + 1]
            fitness_dict = self.calculate_fitness(agent_1, agent_2)
            fitness_1_list = fitness_dict['fitness_1_list']
            fitness_2_list = fitness_dict['fitness_2_list']

            fitnesses += [fitness_1_list, fitness_2_list]

        return fitnesses

    def calculate_fitness(self, agent_1, agent_2, render=False, time_delay=0, measure_specialisation=False,
                          logging=False, logfilename=None):
        """
        Calculates the fitness of a team of agents (composed of two different types of agent). Fitness is calculated
        by running the simulation for t time steps (as specified in the parameter file) with each agent acting every
        time step based on its observations. The simulation is restarted at the end and repeated a certain number of
        times (according to the parameter file). This is done without resetting the random number generator so that
        there are new initial positions for agents and resources in each simulation run.

        @param agent_1: The first Agent object
        @param agent_2: The second Agent object
        @param render: Boolean indicating whether or not simulations will be visualised
        @param time_delay: Integer indicating how many seconds delay (for smoother visualisation)
        @param measure_specialisation: Boolean indicating whether or not specialisation is being measured
        @param logging: Boolean indicating whether or not actions will be logged
        @param logfilename: String name of the file where actions will be logged
        @return: Dictionary containing 'fitness_1_list' (fitnesses of agent 1 for each simulation run), 'fitness_2_list'
        (fitnesses of agent 2 for each simulation run) and 'specialisation_list' (a measure of the degree of
        specialisation observed for the team for each simulation runs)
        """
        # Initialise major variables
        file_reader = None
        fitness_1_list = []
        fitness_2_list = []
        specialisation_list = []
        agent_1_copy = copy.deepcopy(agent_1)
        agent_2_copy = copy.deepcopy(agent_2)

        # Create logging file if logging
        if logging:
            if not logfilename:
                raise RuntimeError("Cannot log results without naming the logfile")

            if not os.path.exists(logfilename):
                file_reader = open(logfilename, "w+")
                header_line = ','.join(f"t={t}" for t in range(self.simulation_length)) + "\n"
                file_reader.write(header_line)
            else:
                file_reader = open(logfilename, "a")

        # Run the simulation several times
        for sim_run in range(self.num_simulation_runs):
            observations = self.env.reset()

            # Initialise variables
            fitness_1 = 0
            fitness_2 = 0
            agent_1_action_list = []
            agent_2_action_list = []

            # Do 1 run of the simulation
            for t in range(self.simulation_length):
                if render:
                    self.env.render()

                robot_actions = []

                for i in range(len(observations)):
                    # TODO: Update this to work with teams greater than 2
                    if i % 2 == 0:
                        robot_actions += [agent_1_copy.act(observations[i])]
                        agent_1_action_list += [robot_actions[-1]]
                    else:
                        robot_actions += [agent_2_copy.act(observations[i])]
                        agent_2_action_list += [robot_actions[-1]]

                # The environment changes according to all their actions
                observations, rewards = self.env.step(robot_actions)

                # Calculate how much of the rewards go to each agent type
                for i in range(len(rewards)):
                    if i % 2 == 0:
                        fitness_1 += rewards[i]
                    else:
                        fitness_2 += rewards[i]

                if time_delay > 0:
                    time.sleep(time_delay)

            # Update averages and seed
            fitness_1_list += [fitness_1]
            fitness_2_list += [fitness_2]

            # Reset agent networks
            agent_1_copy = copy.deepcopy(agent_1)
            agent_2_copy = copy.deepcopy(agent_2)

            # Extra computations if calculating specialisation or logging actions
            if measure_specialisation:
                specialisation_list += [self.env.calculate_ferrante_specialisation()]

            if logging:
                agent_1_action_string = ','.join(str(action) for action in agent_1_action_list) + '\n'
                agent_2_action_string = ','.join(str(action) for action in agent_2_action_list) + '\n'
                file_reader.write(agent_1_action_string)
                file_reader.write(agent_2_action_string)

        if logging:
            file_reader.close()

        return {"fitness_1_list": fitness_1_list, "fitness_2_list": fitness_2_list,
                "specialisation_list": specialisation_list}

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

