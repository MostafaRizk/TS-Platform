import numpy as np

from learning.rwg_centralised import CentralisedRWGLearner
from operator import add


class FullyCentralisedRWGLearner(CentralisedRWGLearner):
    def __init__(self, calculator):
        super().__init__(calculator)

    def learn(self, logging=True):
        # Create population of random genomes
        genome_population = self.create_genome_population()

        # Convert genomes to controllers
        controller_population = self.convert_genomes_to_controllers(genome_population)

        # Get fitnesses of agents
        agent_fitness_lists, team_specialisations = self.fitness_calculator.calculate_fitness_of_agent_population(controller_population, self.calculate_specialisation)
        agent_fitness_average = [np.mean(fitness_list) for fitness_list in agent_fitness_lists]

        best_genome = None
        best_fitness = None

        if self.reward_level != "team":
            raise RuntimeError("Fully centralised teams must use team rewards")

        # Find genome of best team
        else:
            best_team_fitness, best_team_index = float('-inf'), None

            for i in range(0, len(agent_fitness_lists) - 1, self.num_agents):
                team_fitness_list = [0] * len(agent_fitness_lists[i])

                # Get the average team fitness
                for j in range(self.num_agents):
                    team_fitness_list = list(map(add, team_fitness_list, agent_fitness_lists[i + j]))

                team_fitness_average = np.mean(team_fitness_list)

                if team_fitness_average > best_team_fitness:
                    best_team_fitness = team_fitness_average
                    best_team_index = i // self.num_agents

            best_genome = controller_population[best_team_index].get_genome()
            best_fitness = best_team_fitness

        if logging:
            # Save best genome as a model file
            model_name = self.generate_model_name(best_fitness)
            self.save_genome(best_genome, model_name)

            # Log all generated genomes and their fitnesses in one file
            genome_fitness_lists = self.get_genome_fitnesses_from_agent_fitnesses(agent_fitness_lists)
            self.log_all_genomes(genome_population, genome_fitness_lists, team_specialisations)

        return best_genome, best_fitness

    # Helpers ---------------------------------------------------------------------------------------------------------

    def create_genome_population(self):
        genome_population = []

        # Calculate how many genomes to sample based on agent population size, team type and reward level
        num_genomes = 0
        agent_population_size = self.parameter_dictionary['algorithm']['agent_population_size']

        assert agent_population_size % 2 == 0, "Agent population needs to be even"
        num_genomes = agent_population_size // self.num_agents

        # Sample genomes from a normal distribution
        if self.parameter_dictionary['algorithm']['rwg']['sampling_distribution'] == "normal":
            genome_population = self.sample_from_normal(num_genomes)

        # Sample genomes from a uniform distribution
        elif self.parameter_dictionary['algorithm']['rwg']['sampling_distribution'] == "uniform":
            genome_population = self.sample_from_uniform(num_genomes)

        # Sample genomes from a latin hypercube
        elif self.parameter_dictionary['algorithm']['rwg']['sampling_distribution'] == "lhs":
            genome_population = self.sample_from_latin_hypercube(num_genomes)

        if num_genomes == 1:
            genome_population = np.array([genome_population])

        return genome_population