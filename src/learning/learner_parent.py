import json
from fitness import FitnessCalculator
from agents.nn_agent import NNAgent


class Learner:
    def __init__(self, calculator):
        assert isinstance(calculator, FitnessCalculator), "Did not pass a fitness calculator object"

        self.parameter_filename = calculator.get_parameter_filename()
        self.parameter_dictionary = calculator.get_parameter_dictionary()

        self.fitness_calculator = calculator
        self.population_size = self.parameter_dictionary['algorithm']['agent_population_size']
        self.team_type = self.parameter_dictionary['general']['team_type']
        self.reward_level = self.parameter_dictionary['general']['reward_level']
        self.genome_length = self.get_genome_length()

    def learn(self):
        pass

    # Helpers ---------------------------------------------------------------------------------------------------------

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

            # Genomes for heterogeneous teams rewarded at the team level are twice as long because two agent genomes
            # must be concatenated into a larger one
            if self.parameter_dictionary['general']['team_type'] == "heterogeneous" and \
                    self.parameter_dictionary['general']['reward_level'] == "team":
                return num_weights * 2
            else:
                return num_weights

    def convert_genomes_to_agents(self, genome_population):
        """
        Convert list of genomes into list of agents that use those genomes, according to team type and reward level
        specified in the parameters

        @param genome_population: List of genomes
        @return: List of Agent objects that can be used in the environment
        """
        Agent = None
        agent_population = []

        if self.parameter_dictionary['general']['agent_type'] == "nn":
            Agent = NNAgent

        # In homogeneous teams, each genome is used for two identical agents
        if self.team_type == "homogeneous":
            for genome in genome_population:
                agent_1 = Agent(self.fitness_calculator.get_observation_size(), self.fitness_calculator.get_action_size(), self.parameter_filename, genome)
                agent_2 = Agent(self.fitness_calculator.get_observation_size(), self.fitness_calculator.get_action_size(), self.parameter_filename, genome)
                agent_population += [agent_1, agent_2]

        elif self.team_type == "heterogeneous":
            # In heterogeneous teams rewarded at the team level, each genome is two concatenated agents
            if self.reward_level == "team":
                for genome in genome_population:
                    mid = int(len(genome) / 2)
                    genome_part_1 = genome[0:mid]
                    genome_part_2 = genome[mid:]
                    agent_1 = Agent(self.fitness_calculator.get_observation_size(), self.fitness_calculator.get_action_size(), self.parameter_filename, genome_part_1)
                    agent_2 = Agent(self.fitness_calculator.get_observation_size(), self.fitness_calculator.get_action_size(), self.parameter_filename, genome_part_2)
                    agent_population += [agent_1, agent_2]

            # In heterogeneous teams rewarded at the individual level, each genome is a unique agent
            elif self.reward_level == "individual":
                for genome in genome_population:
                    agent_1 = Agent(self.fitness_calculator.get_observation_size(), self.fitness_calculator.get_action_size(), self.parameter_filename, genome)
                    agent_population += [agent_1]

        return agent_population
