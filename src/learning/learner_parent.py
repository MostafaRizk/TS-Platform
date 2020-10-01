import json
from fitness import FitnessCalculator
from agents.nn_agent_lean import NNAgent
from operator import add


class Learner:
    def __init__(self, calculator):
        assert isinstance(calculator, FitnessCalculator), "Did not pass a fitness calculator object"

        self.parameter_filename = calculator.get_parameter_filename()
        self.parameter_dictionary = calculator.get_parameter_dictionary()

        self.fitness_calculator = calculator
        self.population_size = self.parameter_dictionary['algorithm']['agent_population_size']
        environment_name = self.parameter_dictionary['general']['environment']
        self.num_agents = self.parameter_dictionary['environment'][environment_name]['num_agents']
        self.team_type = self.parameter_dictionary['general']['team_type']
        self.reward_level = self.parameter_dictionary['general']['reward_level']
        self.genome_length = self.get_genome_length()
        self.Agent = None

        if self.parameter_dictionary['general']['agent_type'] == "nn":
            self.Agent = NNAgent

    def learn(self, logging):
        pass

    # Helpers ---------------------------------------------------------------------------------------------------------

    def get_parameter_dictionary(self):
        return self.parameter_dictionary

    def get_genome_length(self):
        """
        Get the length of the typical genome that will be used in learning

        @param parameter_filename: Name of the file containing the experiment parameters
        @return: Integer representing the genome length
        """
        if self.parameter_dictionary['general']['agent_type'] == "nn":

            # Get the number of weights in a neural network-based agent
            num_weights = NNAgent(self.fitness_calculator.get_observation_size(),
                                  self.fitness_calculator.get_action_size(), self.parameter_filename).get_num_weights()

            # Genomes for heterogeneous teams rewarded at the team level are longer because multiple agent genomes
            # must be concatenated into a larger one
            if self.parameter_dictionary['general']['team_type'] == "heterogeneous" and \
                    self.parameter_dictionary['general']['reward_level'] == "team":
                return num_weights * self.num_agents
            else:
                return num_weights

    def convert_genomes_to_agents(self, genome_population):
        """
        Convert list of genomes into list of agents that use those genomes, according to team type and reward level
        specified in the parameters

        @param genome_population: List of genomes
        @return: List of Agent objects that can be used in the environment
        """
        agent_population = []

        # In homogeneous teams, each genome is used for two identical agents
        if self.team_type == "homogeneous":
            for genome in genome_population:
                agent = self.Agent(self.fitness_calculator.get_observation_size(),
                                     self.fitness_calculator.get_action_size(), self.parameter_filename, genome)
                agent_population += [agent]

        elif self.team_type == "heterogeneous":
            # In heterogeneous teams rewarded at the team level, each genome is two concatenated agents
            if self.reward_level == "team":
                for genome in genome_population:
                    sub_genome_length = int(len(genome) / self.num_agents)

                    for i in range(self.num_agents):
                        start_index = i * sub_genome_length
                        end_index = (i+1) * sub_genome_length
                        sub_genome = genome[start_index:end_index]

                        agent = self.Agent(self.fitness_calculator.get_observation_size(),
                                         self.fitness_calculator.get_action_size(), self.parameter_filename,
                                         sub_genome)
                        agent_population += [agent]

            # In heterogeneous teams rewarded at the individual level, each genome is a unique agent
            elif self.reward_level == "individual":
                for genome in genome_population:
                    agent = self.Agent(self.fitness_calculator.get_observation_size(),
                                         self.fitness_calculator.get_action_size(), self.parameter_filename, genome)
                    agent_population += [agent]

        return agent_population

    def get_genome_fitnesses_from_agent_fitnesses(self, agent_fitness_lists):
        """
        Given a list of fitness lists of teams of agents, returns the fitness lists of the genomes they came from, based on the
        configuration of team type and reward level

        @param agent_fitnesses: A list of fitness lists for each agent in the agent population
        @return: A list of fitness lists for each genome that the agents came from
        """
        genome_fitness_lists = []

        if self.reward_level == "team":
            for i in range(0, len(agent_fitness_lists) - 1, self.num_agents):
                team_fitness_list = [0] * len(agent_fitness_lists[i])

                for j in range(self.num_agents):
                    team_fitness_list = list(map(add, team_fitness_list, agent_fitness_lists[i+j]))

                genome_fitness_lists += [team_fitness_list]

            return genome_fitness_lists

        elif self.reward_level == "individual" and self.team_type == "heterogeneous":
            return agent_fitness_lists

        else:
            raise RuntimeError('Homogeneous-Individual configuration not fully supported yet')

    def generate_model_name(self, fitness, agent_rank):
        pass

    def save_genome(self, genome, filename):
        """
        Save a genome with under a particular filename

        @param genome: Genome to be saved
        @param filename: String representing the filename to save the genome as
        @return:
        """
        self.Agent.save_given_model(genome, filename)

    @staticmethod
    def get_core_params_in_model_name(parameter_dictionary):
        """
        Given a parameter dictionary, returns a list of the important parameter values that are not specific to the
        algorithm being used (e.g. environment params, seed, agent type etc).

        Note: Tightly coupled with:
        get_seed_genome() in cma.py

        @param parameter_dictionary: Dictionary of parameters
        @return: List of parameter values
        """
        parameters_in_name = []
        # Get general params
        parameters_in_name += [parameter_dictionary['general']['algorithm_selected']]
        parameters_in_name += [parameter_dictionary['general']['team_type']]
        parameters_in_name += [parameter_dictionary['general']['reward_level']]
        parameters_in_name += [parameter_dictionary['general']['agent_type']]
        parameters_in_name += [parameter_dictionary['general']['environment']]
        parameters_in_name += [parameter_dictionary['general']['seed']]

        # Get environment params
        parameters_in_name += [parameter_dictionary['environment']['slope']['num_agents']]
        parameters_in_name += [parameter_dictionary['environment']['slope']['num_resources']]
        parameters_in_name += [parameter_dictionary['environment']['slope']['sensor_range']]
        parameters_in_name += [parameter_dictionary['environment']['slope']['sliding_speed']]
        parameters_in_name += [parameter_dictionary['environment']['slope']['arena_length']]
        parameters_in_name += [parameter_dictionary['environment']['slope']['arena_width']]
        parameters_in_name += [parameter_dictionary['environment']['slope']['cache_start']]
        parameters_in_name += [parameter_dictionary['environment']['slope']['slope_start']]
        parameters_in_name += [parameter_dictionary['environment']['slope']['source_start']]
        parameters_in_name += [parameter_dictionary['environment']['slope']['base_cost']]
        parameters_in_name += [parameter_dictionary['environment']['slope']['upward_cost_factor']]
        parameters_in_name += [parameter_dictionary['environment']['slope']['downward_cost_factor']]
        parameters_in_name += [parameter_dictionary['environment']['slope']['carry_factor']]
        parameters_in_name += [parameter_dictionary['environment']['slope']['resource_reward']]
        parameters_in_name += [parameter_dictionary['environment']['slope']['episode_length']]
        parameters_in_name += [parameter_dictionary['environment']['slope']['num_episodes']]

        # Get agent params for relevant agent type
        agent_type = parameter_dictionary['general']['agent_type']

        if agent_type == 'nn':
            parameters_in_name += [parameter_dictionary['agent']['nn']['architecture']]
            parameters_in_name += [parameter_dictionary['agent']['nn']['bias']]
            parameters_in_name += [parameter_dictionary['agent']['nn']['hidden_layers']]
            parameters_in_name += [parameter_dictionary['agent']['nn']['hidden_units_per_layer']]
            parameters_in_name += [parameter_dictionary['agent']['nn']['activation_function']]

        return parameters_in_name

    @staticmethod
    def get_additional_params_in_model_name(parameter_dictionary):
        pass

    @staticmethod
    def get_results_headings(parameter_dictionary):
        """
        Given a parameter dictionary, returns a list of the names of important parameters that are not specific to the
        algorithm being used

        @param parameter_dictionary:
        @return:
        """
        # TODO: Change the heading so it accommodates other agent types
        headings = ["algorithm_selected",
                    "team_type",
                    "reward_level",
                    "agent_type",
                    "environment",
                    "seed",
                    "num_agents",
                    "num_resources",
                    "sensor_range",
                    "sliding_speed",
                    "arena_length",
                    "arena_width",
                    "cache_start",
                    "slope_start",
                    "source_start",
                    "base_cost",
                    "upward_cost_factor",
                    "downward_cost_factor",
                    "carry_factor",
                    "resource_reward",
                    "episode_length",
                    "num_episodes",
                    "architecture",
                    "bias",
                    "hidden_layers",
                    "hidden_units_per_layer",
                    "activation_function"]

        return headings
