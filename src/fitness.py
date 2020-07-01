import json
import os
import numpy as np
import time
from envs.slope import SlopeEnv


class FitnessCalculator:

    def __init__(self, parameter_filename):

        if parameter_filename is None:
            raise RuntimeError("No parameter file specified for the fitness function")

        parameter_dictionary = json.loads(open(parameter_filename).read())

        self.env = SlopeEnv(parameter_filename)

        # Get size of input and output space (for use in agent creation)
        self.observation_size = self.env.get_observation_size()
        self.action_size = self.env.get_action_size()

        # Seeding values
        self.random_seed = parameter_dictionary['general']['seed']
        self.np_random = np.random.RandomState(self.random_seed)

        self.simulation_length = parameter_dictionary['environment']['simulation_length']

        self.num_simulation_runs = parameter_dictionary['environment']['num_simulation_runs']

    def calculate_fitness_of_population(self, population):
        """
        Takes a population of Agent objects, places each pair on a team and calculates the fitnesses of each

        @param population: List of Agent objects
        @return: List containing fitness value of each Agent in the population
        """

        fitnesses = []

        for i in range(0, len(population), 2):
            agent_1 = population[i]
            agent_2 = population[i + 1]
            fitness_1, fitness_2 = self.calculate_fitness(agent_1, agent_2)

            fitnesses += [fitness_1, fitness_2]

        return fitnesses

    def calculate_fitness(self, agent_type_1, agent_type_2, render=False, time_delay=0, measure_specialisation=False,
                          logging=False, logfilename=None):
        """
        Calculates the fitness of a team of agents (composed of two different types of agent). Fitness is calculated
        by running the simulation for t time steps (as specified in the parameter file) with each agent acting every
        time step based on its observations. The simulation is reset at the end and repeated a certain number of
        times (according to the parameter file) each with a different temporary seed. The average is taken over all
        runs of the simulation.

        @param agent_type_1: The first Agent object
        @param agent_type_2: The second Agent object
        @param render: Boolean indicating whether or not simulations will be visualised
        @param time_delay: Integer indicating how many seconds delay (for smoother visualisation)
        @param measure_specialisation: Boolean indicating whether or not specialisation is being measured
        @param logging: Boolean indicating whether or not actions will be logged
        @param logfilename: String name of the file where actions will be logged
        @return: Dictionary containing 'fitness_1' (agent1 fitness), 'fitness_2' (agent 2 fitness) and 'specialisation'
        (a measure of the degree of specialisation observed)
        """
        # Initialise major variables
        temp_seed = self.random_seed
        file_reader = None
        average_fitness_1 = 0
        average_fitness_2 = 0
        average_specialisation = 0

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
                        robot_actions += [agent_type_1.act(observations[i])]
                        agent_1_action_list += [robot_actions[-1]]
                    else:
                        robot_actions += [agent_type_2.act(observations[i])]
                        agent_2_action_list += [robot_actions[-1]]

                # The environment changes according to all their actions
                observations, rewards = self.env.step(robot_actions)

                fitness_1 += rewards[0]
                fitness_2 += rewards[1]

                if time_delay > 0:
                    time.sleep(time_delay)

            # Update averages and seed
            average_fitness_1 += fitness_1
            average_fitness_2 += fitness_2
            temp_seed += 1  # TODO: Check that the logic of incrementing the seed every run makes sense

            # Extra computations if calculating specialisation or logging actions
            if measure_specialisation:
                average_specialisation += self.env.calculate_ferrante_specialisation()

            if logging:
                agent_1_action_string = ','.join(str(action) for action in agent_1_action_list) + '\n'
                agent_2_action_string = ','.join(str(action) for action in agent_2_action_list) + '\n'
                file_reader.write(agent_1_action_string)
                file_reader.write(agent_2_action_string)

        average_fitness_1 /= self.num_simulation_runs
        average_fitness_2 /= self.num_simulation_runs

        if measure_specialisation:
            average_specialisation /= self.num_simulation_runs
        else:
            average_specialisation = None

        if logging:
            file_reader.close()

        return {"fitness_1": average_fitness_1, "fitness_2": average_fitness_2,
                "specialisation": average_specialisation}

    # Helpers ---------------------------------------------------------------------------------------------------------

    def get_observation_size(self):
        return self.observation_size

    def get_action_size(self):
        return self.action_size

    def get_rng(self):
        return self.np_random
