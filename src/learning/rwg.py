from learning.learner_parent import Learner
from scipy.stats import multivariate_normal
import numpy as np


class RWGLearner(Learner):
    def learn(self):
        """
        Find the best genome and its fitness according to the rwg algorithm and the set parameters

        @return: The genome with the highest fitness and its fitness
        """
        # Create population of random genomes
        genome_population = self.create_genome_population()

        # Convert genomes to agents
        agent_population = self.convert_genomes_to_agents(genome_population)

        # Get fitnesses of agents
        agent_fitnesses = self.fitness_calculator.calculate_fitness_of_agent_population(agent_population)

        # Find genome of best agent
        if self.reward_level == "individual":
            best_fitness, best_fitness_index = max([(value, index) for index, value in enumerate(agent_fitnesses)])
            best_agent = agent_population[best_fitness_index]
            best_genome = best_agent.get_genome()
            return best_genome, best_fitness

        # Find genome of best team
        elif self.reward_level == "team":
            best_team_fitness, best_team_index = float('-inf'), None

            for i in range(0, len(agent_fitnesses)-1, 2):
                team_fitness = agent_fitnesses[i] + agent_fitnesses[i+1]

                if team_fitness > best_team_fitness:
                    best_team_fitness = team_fitness
                    best_team_index = i

            # For homogeneous teams with team reward, the genome of one agent is also the team genome
            if self.team_type == "homogeneous":
                best_genome = agent_population[best_team_index].get_genome()

            # For heterogeneous teams with team reward, the genome of the team is the genome of both agents concatenated
            elif self.team_type == "heterogeneous":
                genome_part_1 = agent_population[best_team_index].get_genome()
                genome_part_2 = agent_population[best_team_index+1].get_genome()
                best_genome = np.concatenate([genome_part_1, genome_part_2])

            return best_genome, best_team_fitness

    # Helpers ---------------------------------------------------------------------------------------------------------

    def create_genome_population(self):
        """
        Create a list of genomes, randomly sampled according to the experiment parameters

        @return: List of genomes
        """
        genome_population = []

        # Calculate how many genomes to sample based on agent population size, team type and reward level
        num_genomes = 0
        agent_population_size = self.parameter_dictionary['algorithm']['agent_population_size']

        if self.team_type == "heterogeneous" and self.reward_level == "individual":
            num_genomes = agent_population_size
        else:
            assert agent_population_size % 2 == 0, "Agent population needs to be even"
            num_genomes = agent_population_size/2

        # Sample genomes from a normal distribution
        if self.parameter_dictionary['algorithm']['rwg']['sampling_distribution'] == "normal":
            genome_population = self.sample_from_normal(num_genomes)

        # Sample genomes from a uniform distribution
        elif self.parameter_dictionary['algorithm']['rwg']['sampling_distribution'] == "uniform":
            genome_population = self.sample_from_uniform(num_genomes)

        return genome_population

    def sample_from_normal(self, num_genomes):
        """
        Sample genomes from a normal distribution

        @param num_genomes: Number of genomes to sample
        @return: Sampled genomes
        """
        mean = self.parameter_dictionary['algorithm']['rwg']['sampling_distribution']['normal']['mean']
        std = self.parameter_dictionary['algorithm']['rwg']['sampling_distribution']['normal']['std']
        seed = self.parameter_dictionary['general']['seed']

        mean_array = [mean] * self.genome_length
        random_variable = multivariate_normal(mean=mean_array, cov=np.identity(self.genome_length) * std)
        return random_variable.rvs(num_genomes, seed)

    def sample_from_uniform(self, num_genomes):
        """
        Sample genomes from a uniform distribution

        @param num_genomes: Number of genomes to sample
        @return: Sampled genomes
        """
        min = self.parameter_dictionary['algorithm']['rwg']['sampling_distribution']['uniform']['min']
        max = self.parameter_dictionary['algorithm']['rwg']['sampling_distribution']['uniform']['max']
        seed = self.parameter_dictionary['general']['seed']

        min_array = np.full((1, self.genome_length), min)
        max_array = np.full((1, self.genome_length), max)
        random_state = np.random.RandomState(seed)
        return random_state.uniform(min_array, max_array, (num_genomes, self.genome_length))
