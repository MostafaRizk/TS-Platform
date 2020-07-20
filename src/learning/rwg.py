from learning.learner_parent import Learner
from scipy.stats import multivariate_normal
import numpy as np


class RWGLearner(Learner):
    def __init__(self, calculator):
        super().__init__(calculator)

        if self.parameter_dictionary['general']['algorithm_selected'] != "rwg":
            raise RuntimeError(f"Cannot run rwg. Parameters request "
                               f"{self.parameter_dictionary['general']['algorithm_selected']}")

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
        agent_fitness_lists = self.fitness_calculator.calculate_fitness_of_agent_population(agent_population)
        agent_fitness_average = [sum(fitness_list)/len(fitness_list) for fitness_list in agent_fitness_lists]

        best_genome = None
        best_fitness = None

        # Find genome of best agent
        if self.reward_level == "individual":
            best_fitness, best_fitness_index = max([(value, index) for index, value in enumerate(agent_fitness_average)])
            best_agent = agent_population[best_fitness_index]
            best_genome = best_agent.get_genome()

        # Find genome of best team
        elif self.reward_level == "team":
            best_team_fitness, best_team_index = float('-inf'), None

            for i in range(0, len(agent_fitness_lists) - 1, 2):
                # Get the average team fitness
                fitness_1_list = agent_fitness_lists[i]
                fitness_2_list = agent_fitness_lists[i+1]
                zipped_fitness = zip(fitness_1_list, fitness_2_list)
                team_fitness_list = [fitness_1 + fitness_2 for (fitness_1, fitness_2) in zipped_fitness]
                team_fitness_average = sum(team_fitness_list)/len(team_fitness_list)

                if team_fitness_average > best_team_fitness:
                    best_team_fitness = team_fitness_average
                    best_team_index = i

            # For homogeneous teams with team reward, the genome of one agent is also the team genome
            if self.team_type == "homogeneous":
                best_genome = agent_population[best_team_index].get_genome()

            # For heterogeneous teams with team reward, the genome of the team is the genome of both agents concatenated
            elif self.team_type == "heterogeneous":
                genome_part_1 = agent_population[best_team_index].get_genome()
                genome_part_2 = agent_population[best_team_index + 1].get_genome()
                best_genome = np.concatenate([genome_part_1, genome_part_2])

            best_fitness = best_team_fitness

        # Save best genome as a model file
        model_name = self.generate_model_name(best_fitness)
        self.save_genome(best_genome, model_name)

        # Log all generated genomes and their fitnesses in one file
        genome_fitness_lists = self.get_genome_fitnesses_from_agent_fitnesses(agent_fitness_lists)
        self.log_all_genomes(genome_population, genome_fitness_lists)

        return best_genome, best_fitness

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
            num_genomes = agent_population_size // 2

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
        mean = self.parameter_dictionary['algorithm']['rwg']['normal']['mean']
        std = self.parameter_dictionary['algorithm']['rwg']['normal']['std']
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
        min = self.parameter_dictionary['algorithm']['rwg']['uniform']['min']
        max = self.parameter_dictionary['algorithm']['rwg']['uniform']['max']
        seed = self.parameter_dictionary['general']['seed']

        min_array = np.full((1, self.genome_length), min)
        max_array = np.full((1, self.genome_length), max)
        random_state = np.random.RandomState(seed)
        return random_state.uniform(min_array, max_array, (num_genomes, self.genome_length))

    def log_all_genomes(self, genomes, genome_fitness_list):
        assert len(genomes) == len(genome_fitness_list), "List of genomes and list of fitnesses are unequal"

        parameters_in_name = ["all_genomes"]
        parameters_in_name += Learner.get_core_params_in_model_name(self.parameter_dictionary)
        parameters_in_name += RWGLearner.get_additional_params_in_model_name(self.parameter_dictionary)
        parameters_in_name += [".csv"]
        filename = "_".join([str(param) for param in parameters_in_name])

        f = open(filename, "w")

        for i in range(len(genomes)):
            genome = genomes[i]
            fitness_list = genome_fitness_list[i]

            genome_str = str(genome.tolist()).strip("[]") + "," + ",".join(str(fitness) for fitness in fitness_list) + "\n"
            f.write(genome_str)

        f.close()

    def generate_model_name(self, fitness):
        """
        Create a name string for a model generated using the given parameter file and fitness value

        @param fitness: Fitness of the model to be saved
        @return:
        """
        parameters_in_name = Learner.get_core_params_in_model_name(self.parameter_dictionary)
        parameters_in_name += RWGLearner.get_additional_params_in_model_name(self.parameter_dictionary)

        # Get fitness
        parameters_in_name += [fitness]

        # Put all in a string and return
        return "_".join([str(param) for param in parameters_in_name])

    @staticmethod
    def get_additional_params_in_model_name(parameter_dictionary):
        parameters_in_name = []

        # Get algorithm params for relevant algorithm & distribution
        parameters_in_name += [parameter_dictionary['algorithm']['agent_population_size']]
        sampling_distribution = parameter_dictionary['algorithm']['rwg']['sampling_distribution']
        parameters_in_name += [sampling_distribution]

        if sampling_distribution == "normal":
            parameters_in_name += [parameter_dictionary['algorithm']['rwg']['normal']['mean']]
            parameters_in_name += [parameter_dictionary['algorithm']['rwg']['normal']['std']]

        elif sampling_distribution == "uniform":
            parameters_in_name += [parameter_dictionary['algorithm']['rwg']['uniform']['min']]
            parameters_in_name += [parameter_dictionary['algorithm']['rwg']['uniform']['max']]

        return parameters_in_name
