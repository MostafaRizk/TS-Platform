import copy
import json
import os
import numpy as np
import time
from envs.slope import SlopeEnv
from envs.tmaze import TMazeEnv
from envs.box_pushing import BoxPushingEnv


class FitnessCalculator:

    def __init__(self, parameter_filename):

        if parameter_filename is None:
            raise RuntimeError("No parameter file specified for the fitness function")

        self.parameter_filename = parameter_filename
        self.parameter_dictionary = json.loads(open(parameter_filename).read())

        if self.parameter_dictionary["general"]["environment"] == "slope":
            self.env = SlopeEnv(parameter_filename)
        elif self.parameter_dictionary["general"]["environment"] == "tmaze":
            self.env = TMazeEnv(parameter_filename)
        elif self.parameter_dictionary["general"]["environment"] == "box_pushing":
            self.env = BoxPushingEnv(parameter_filename)

        environment_name = self.parameter_dictionary['general']['environment']
        self.num_agents = self.parameter_dictionary['environment'][environment_name]['num_agents']

        # Get size of input and output space (for use in agent creation)
        self.observation_size = self.env.get_observation_size()
        self.action_size = self.env.get_action_size()

        # Seeding values
        self.random_seed = self.parameter_dictionary['general']['seed']
        self.np_random = np.random.RandomState(self.random_seed)

        #self.episode_length = self.parameter_dictionary['environment'][environment_name]['episode_length']
        self.episode_length = self.env.get_episode_length()
        self.num_episodes = self.parameter_dictionary['environment'][environment_name]['num_episodes']

        self.learning_type = self.parameter_dictionary['general']['learning_type']

    def calculate_fitness_of_agent_population(self, population, calculate_specialisation):
        """
        Takes a population of controller objects, if each controller is an agent of the environment then places each
        group of agents in a team and calculates the fitnesses of each. If each controller is a team, then calculates
        the fitness of each agent on that team

        @param population: List of Agent objects
        @return: List containing fitness value of each agent
        """

        if self.learning_type != "fully-centralised":
            assert len(population) % self.num_agents == 0, "Population needs to be divisible by the number of agents per team"

        fitnesses = []
        specialisations = []
        participations = []
        behaviour_characterisations = []
        trajectories = []
        agents_per_team = self.num_agents

        if self.learning_type != "fully-centralised":
            for i in range(0, len(population), agents_per_team):
                agent_list = [population[i+j] for j in range(0, agents_per_team)]
                results_dict = self.calculate_fitness(agent_list, measure_specialisation=calculate_specialisation)
                fitness_matrix = results_dict['fitness_matrix']

                # Specialisation of the team calculated through several measures. List of lists. One list of measures for each episode
                specialisation_measures = results_dict["specialisation_list"]

                behaviour_characterisation_matrix = results_dict['behaviour_characterisation_matrix']

                fitnesses += fitness_matrix
                specialisations += [specialisation_measures]
                participations += results_dict["participation_list"]
                behaviour_characterisations += behaviour_characterisation_matrix
                trajectories += results_dict["trajectory_matrix"]

        else:
            for controller in population:
                results_dict = self.calculate_fitness([controller], measure_specialisation=calculate_specialisation)
                fitness_matrix = results_dict['fitness_matrix']
                specialisation_measures = results_dict["specialisation_list"]
                behaviour_characterisation_matrix = results_dict['behaviour_characterisation_matrix']

                fitnesses += fitness_matrix
                specialisations += [specialisation_measures]
                participations += results_dict["participation_list"]
                behaviour_characterisations += behaviour_characterisation_matrix
                trajectories += results_dict["trajectory_matrix"]

        return fitnesses, specialisations, participations, behaviour_characterisations, trajectories

    def calculate_fitness(self, controller_list, render=False, time_delay=0, measure_specialisation=False,
                          logging=False, logfilename=None, render_mode="human"):
        """
        Calculates the fitness of a team of agents. Fitness is calculated
        by running the simulation for t time steps (as specified in the parameter file) with each agent acting every
        time step based on its observations. The simulation is restarted at the end and repeated a certain number of
        times (according to the parameter file). This is done without resetting the random number generator so that
        there are new initial positions for agents and resources in each simulation run.

        @param controller_list: List of controller objects. One per agent unless fully centralised.
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

        if self.learning_type != "fully-centralised":
            assert len(controller_list) == self.num_agents, "Agents passed to function do not match parameter file"
        else:
            assert len(controller_list) == 1, "Must only have 1 controller for fully centralised learning"

        # Initialise major variables
        file_reader = None
        fitness_matrix = [[0]*self.num_episodes for _ in range(self.num_agents)]
        specialisation_list = []
        participation_list = []
        total_resources_per_episode = [0]*self.num_episodes

        # For each agent, contains its BC value in every episode
        # Concatenate if evaluating team novelty
        behaviour_characterisation_matrix = [[] for _ in range(self.num_agents)]

        # Trajectory of each
        trajectory_matrix = [[[None for _ in range(self.episode_length)] for _ in range(self.num_episodes)] for _ in range(self.num_agents)]

        controller_copies = [copy.deepcopy(controller) for controller in controller_list]
        video_frames = [None]*self.num_episodes
        self.env.reset_rng()

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
            agent_action_matrix = [[-1]*self.episode_length for _ in range(self.num_agents)]
            current_episode_reward_matrix = [[0]*self.episode_length for _ in range(self.num_agents)]
            frames = []

            # Do 1 run of the simulation
            for t in range(self.episode_length):
                if render and render_mode == "rgb_array":
                    frames += [self.env.render(mode=render_mode)]
                elif render:
                    self.env.render(mode=render_mode)

                agent_actions = []

                if self.learning_type != "fully-centralised":
                    for i in range(len(observations)):
                        agent_actions += [controller_copies[i].act(observations[i])]
                        agent_action_matrix[i][t] = [agent_actions[-1]]

                else:
                    full_observation = []

                    for obs in observations:
                        full_observation = np.append(full_observation, obs)

                    agent_actions = controller_copies[0].act(full_observation, self.num_agents)

                    for i in range(len(observations)):
                        agent_action_matrix[i][t] = [agent_actions[i]]

                # The environment changes according to all their actions
                observations, rewards = self.env.step(agent_actions)

                # Calculate how much of the rewards go to each agent type
                for i in range(len(rewards)):
                    fitness_matrix[i][episode] += rewards[i]
                    current_episode_reward_matrix[i][t] = rewards[i]

                if time_delay > 0:
                    time.sleep(time_delay)

            # Reset agent networks
            controller_copies = [copy.deepcopy(controller) for controller in controller_list]

            # Extra computations if calculating specialisation or logging actions
            if measure_specialisation:
                specialisation_list += [self.env.calculate_specialisation()]

            participation_list += [self.env.calculate_participation()]
            bc_for_agent = self.env.get_behaviour_characterisation()
            trajectories = self.env.get_trajectories()
            total_resources_per_episode[episode] = self.env.get_total_resources_retrieved()
            video_frames[episode] = np.array(frames)

            for i in range(self.num_agents):
                behaviour_characterisation_matrix[i] += [bc_for_agent[i]]
                trajectory_matrix[i][episode] = trajectories[i]

            if logging:
                #for agent_action_list in agent_action_matrix:
                #    action_string = ','.join(str(action) for action in agent_action_list) + '\n'
                #    file_reader.write(action_string)
                for reward_list in current_episode_reward_matrix:
                    reward_string = ','.join(str(reward) for reward in reward_list) + '\n'
                    file_reader.write(reward_string)

        if logging:
            file_reader.close()

        return {"fitness_matrix": fitness_matrix, "specialisation_list": specialisation_list, "participation_list": participation_list, "video_frames": video_frames, "behaviour_characterisation_matrix": behaviour_characterisation_matrix, "trajectory_matrix": trajectory_matrix, "total_resources": total_resources_per_episode}

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

