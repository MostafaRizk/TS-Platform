import numpy as np

from learning.learner_centralised import CentralisedLearner
from learning.ga_parent import GALearner
from deap import base, creator, algorithms, tools


class CentralisedGALearner(CentralisedLearner, GALearner):
    def __init__(self, calculator):
        pass

    def learn(self, logging=True):

        seed = self.parameter_dictionary['general']['seed']
        num_generations = self.parameter_dictionary['algorithm']['ga']['generations']
        pop_size = self.get_genome_population_length()
        np_random = np.random.RandomState(seed)
        genome_length = self.get_genome_length()
        mutation_probability = self.parameter_dictionary['algorithm']['ga']['mutation_probability']
        tournament_size = self.parameter_dictionary['algorithm']['ga']['tournament_size']
        mu = self.parameter_dictionary['algorithm']['ga']['mu']
        sigma = self.parameter_dictionary['algorithm']['ga']['sigma']
        num_parents = self.parameter_dictionary['algorithm']['ga']['num_parents']
        num_children = self.parameter_dictionary['algorithm']['ga']['num_children']

        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)
        toolbox = base.Toolbox()
        toolbox.register("attr_float", np_random.random)
        toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=genome_length)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("mate", tools.cxOnePoint)
        toolbox.register("select", tools.selTournament, tournsize=tournament_size)
        toolbox.register("mutate", tools.mutGaussian, mu=mu, sigma=sigma, indpb=mutation_probability)
        pop = toolbox.population(n=pop_size)

        for g in range(num_generations):

            if self.multithreading:
                raise RuntimeError("Multithreading not implemented for GAs")

            else:
                agent_fitness_lists, team_specialisations, agent_bc_vectors = self.fitness_calculator.calculate_fitness_of_agent_population(controller_population, self.calculate_specialisation)

            # Convert agent fitnesses into genome fitnesses
            genome_fitness_lists = self.get_genome_fitnesses_from_agent_fitnesses(agent_fitness_lists)
            genome_fitness_average = [np.mean(fitness_list) for fitness_list in genome_fitness_lists]
            team_bc_vectors = self.get_team_bc_from_agent_bc(agent_bc_vectors)

            if not self.using_novelty:
                pass

            else:
                for ind, fit in zip(pop, scores):
                    ind.fitness.values = fit  # TODO: Score AND validity?

            # Select the next generation individuals
            parents = toolbox.select(pop, num_parents)

            # Clone the selected individuals
            parents = map(toolbox.clone, parents)

            # Crossover
            children = toolbox.mate(parents[0], parents[1])
            del parents[0].fitness.values
            del parents[1].fitness.values

            # Mutate
            for i in range(len(children)):
                if np_random.random() < mutation_probability:
                    toolbox.mutate(children[i])

            # Evaluate children

            # Replace most similar individuals in population

    # Helpers ---------------------------------------------------------------------------------------------------------

    def get_genome_population_length(self):
        """
        Calculates how many genomes should be in the population based on the number of agents

        @return: Integer representing the length of the genome population
        """
        # If teams are heterogneous and rewards are individual, there must be one genome per agent
        if self.team_type == "heterogeneous" and self.reward_level == "individual":
            return self.parameter_dictionary['algorithm']['agent_population_size']

        # For most configurations, each genome is either copied onto multiple agents or split across multiple agents
        else:
            return self.parameter_dictionary['algorithm']['agent_population_size'] / self.num_agents