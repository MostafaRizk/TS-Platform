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
                fitness_1, fitness_2 = self.calculate_fitness(individual_1, individual_2, render)
                team_fitness = fitness_1 + fitness_2
                fitnesses += [team_fitness]

                file_name = f"landscape_analysis_{team_type}_{selection_level}_{self.random_seed}.csv"
                flacco_file = open(file_name, "a")
                genome_str = str(genome.tolist()).strip("[]") + "," + str(team_fitness) + "\n"
                flacco_file.write(genome_str)
                flacco_file.close()

        elif team_type == "heterogeneous" and selection_level == "team":
            for genome in population:
                mid = int(len(genome) / 2)
                individual_1 = genome[0:mid]
                individual_2 = genome[mid:]
                fitness_1, fitness_2 = self.calculate_fitness(individual_1,
                                                              individual_2, render)
                team_fitness = fitness_1 + fitness_2
                fitnesses += [team_fitness]

                file_name = f"landscape_analysis_{team_type}_{selection_level}_{self.random_seed}.csv"
                flacco_file = open(file_name, "a")
                genome_str = str(genome.tolist()).strip("[]") + "," + str(team_fitness) + "\n"
                flacco_file.write(genome_str)
                flacco_file.close()

        elif (team_type == "heterogeneous" and selection_level == "individual") or \
                (team_type == "homogeneous" and selection_level == "individual"):
            for i in range(0, len(population), 2):
                individual_1 = population[i]
                individual_2 = population[i + 1]
                fitness_1, fitness_2 = self.calculate_fitness(individual_1,
                                                              individual_2, render)

                fitnesses += [fitness_1, fitness_2]

                file_name = f"landscape_analysis_{team_type}_{selection_level}_{self.random_seed}.csv"
                flacco_file = open(file_name, "a")
                genome_str_1 = str(individual_1.tolist()).strip("[]") + "," + str(fitness_1) + "\n"
                genome_str_2 = str(individual_2.tolist()).strip("[]") + "," + str(fitness_2) + "\n"
                flacco_file.write(genome_str_1)
                flacco_file.write(genome_str_2)
                flacco_file.close()

        else:
            raise RuntimeError("Invalid team type and/or selection level")

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
            temp_seed += 1

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
