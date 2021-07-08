import numpy as np
import os

from learning.learner_centralised import CentralisedLearner
from learning.ga_parent import GALearner
from deap import base, creator, algorithms, tools
from collections import deque


class CentralisedGALearner(CentralisedLearner, GALearner):
    def __init__(self, calculator):
        super().__init__(calculator)
        self.seed = self.parameter_dictionary['general']['seed']
        self.np_random = np.random.RandomState(self.seed)

    def learn(self, logging=True):
        num_generations = self.parameter_dictionary['algorithm']['ga']['generations']
        pop_size = self.get_genome_population_length()
        genome_length = self.get_genome_length()
        mutation_probability = self.parameter_dictionary['algorithm']['ga']['mutation_probability']
        tournament_size = self.parameter_dictionary['algorithm']['ga']['tournament_size']
        mu = self.parameter_dictionary['algorithm']['ga']['mu']
        sigma = self.parameter_dictionary['algorithm']['ga']['sigma']
        num_parents = self.parameter_dictionary['algorithm']['ga']['num_parents']
        num_children = self.parameter_dictionary['algorithm']['ga']['num_children']
        crowding_factor = self.parameter_dictionary['algorithm']['ga']['crowding_factor']

        if self.using_novelty:
            recent_insertions = deque(maxlen=self.novelty_params['insertions_before_increase'])

        if num_parents != 2:
            raise RuntimeError("Only supports 2 parents for now")

        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)
        toolbox = base.Toolbox()
        toolbox.register("attr_float", self.np_random.random)
        toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=genome_length)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("mate", tools.cxOnePoint)
        toolbox.register("select", tools.selTournament, tournsize=tournament_size)
        toolbox.register("mutate", tools.mutGaussian, mu=mu, sigma=sigma, indpb=mutation_probability)
        genome_population = toolbox.population(n=pop_size)
        controller_population = self.convert_genomes_to_controllers(genome_population)

        if self.multithreading:
            raise RuntimeError("Multithreading not implemented for GAs")

        else:
            agent_fitness_lists, team_specialisations, agent_bc_vectors = self.fitness_calculator.calculate_fitness_of_agent_population(controller_population, self.calculate_specialisation)

        # Convert agent fitnesses into genome fitnesses
        genome_fitness_lists = self.get_genome_fitnesses_from_agent_fitnesses(agent_fitness_lists)
        genome_fitness_average = [np.mean(fitness_list) for fitness_list in genome_fitness_lists]
        team_bc_vectors = self.get_team_bc_from_agent_bc(agent_bc_vectors)

        best_genome = None
        best_genome_fitness = float('-Inf')

        if not self.using_novelty:
            raise RuntimeError("To remove novelty completely, set novelty weight to 0")

        else:
            # Calculate novelty
            scores = genome_fitness_average[:]

            for index in range(len(scores)):
                if scores[index] > best_genome_fitness:
                    best_genome_fitness = scores[index]
                    best_genome = genome_population[index]

            initial_fitness = best_genome_fitness

            for ind, score in zip(genome_population, scores):
                ind.fitness.values = (score,)

            for generation in range(num_generations):
                # Select the next generation individuals
                parents = toolbox.select(genome_population, num_parents)

                # Clone the selected individuals
                parents = list(map(toolbox.clone, parents))

                # Crossover
                toolbox.mate(parents[0], parents[1])
                children = parents
                del children[0].fitness.values
                del children[1].fitness.values

                # Mutate
                for i in range(len(children)):
                    if self.np_random.random() < mutation_probability:
                        toolbox.mutate(children[i])

                # Evaluate children
                child_controllers = self.convert_genomes_to_controllers(children)
                child_agent_fitness_lists, child_team_specialisations, child_agent_bc_vectors = self.fitness_calculator.calculate_fitness_of_agent_population(child_controllers, self.calculate_specialisation)

                # Convert agent fitnesses into genome fitnesses
                child_genome_fitness_lists = self.get_genome_fitnesses_from_agent_fitnesses(child_agent_fitness_lists)
                child_genome_fitness_average = [np.mean(fitness_list) for fitness_list in child_genome_fitness_lists]
                child_team_bc_vectors = self.get_team_bc_from_agent_bc(child_agent_bc_vectors)

                # Calculate novelty
                child_scores = child_genome_fitness_average[:]

                # Replace most similar individuals in population
                indicies_to_replace = []

                for k in range(len(children)):
                    child_bc = child_team_bc_vectors[k]
                    min_distance = float('Inf')
                    loser_index = None

                    for i in range(crowding_factor):
                        j = self.np_random.randint(pop_size)
                        distance = self.calculate_behaviour_distance(a=child_bc, b=team_bc_vectors[j])

                        if distance < min_distance:
                            min_distance = distance
                            loser_index = j

                    indicies_to_replace += [loser_index]

                for i in range(len(indicies_to_replace)):
                    loser_index = indicies_to_replace[i]
                    genome_population[loser_index] = children[i]
                    genome_population[loser_index].fitness.values = (child_scores[i],)

                    if child_scores[i] > best_genome_fitness:
                        best_genome_fitness = child_scores[i]
                        best_genome = children[i]

        print(f"Best fitness is {best_genome_fitness}")

        # Save best model
        model_name = self.generate_model_name(best_genome_fitness)
        self.log(best_genome, best_genome_fitness, "final", initial_fitness)

        # Log evolution details to file
        #log_file_name = model_name + ".log"
        #f = open(log_file_name, "w")
        #f.write(mystdout.getvalue())
        #f.close()

        # Reset output stream
        #sys.stdout = old_stdout

        return best_genome, best_genome_fitness

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
            return self.parameter_dictionary['algorithm']['agent_population_size'] // self.num_agents

    def get_team_bc_from_agent_bc(self, agent_bc_vectors):
        """
        Given a list of behaviour characterisation vectors (one for each agent), concatenates the vectors for all the
        agents on a team so that each team has a vector.

        @param agent_bc_vectors: List of behaviour characterisation vectors (one for each agent)
        @return: List of behaviour characterisation vectors (one for each team)
        """
        team_bc_vectors = []

        if self.reward_level == "team":
            for i in range(0, len(agent_bc_vectors) - 1, self.num_agents):
                team_i_vector = []

                for j in range(self.num_agents):
                    team_i_vector += agent_bc_vectors[i+j]

                team_bc_vectors += [team_i_vector]

            return team_bc_vectors

        else:
            raise RuntimeError('Cannot calculate team behaviour characterisation when rewards are not at the team level')

    def log(self, genome, genome_fitness, generation, initial_fitness):
        """
        Save the genome model and save fitness and parameters to a results file

        @param genome: The genome being logged
        @param genome_fitness: The fitness of the genome
        @param generation: The current generation of the GA
        @param initial_fitness: The best initial fitness
        @return:
        """
        # Save model
        model_name = self.generate_model_name(genome_fitness)
        model_name = f"{model_name}_{generation}"
        self.save_genome(genome, model_name)

        # Log results
        results_filename = f"results_{generation}.csv"
        results_file = None

        if not os.path.isfile(results_filename):
            results_file = open(results_filename, 'a')

            # Write header line of results file
            result_headings = self.get_results_headings(self.parameter_dictionary)
            result_headings += ["initial_fitness", "fitness", "model_name"]
            result_headings = ",".join(result_headings)
            result_headings += "\n"
            results_file.write(result_headings)

        else:
            results_file = open(results_filename, 'a')

        result_data = CentralisedLearner.get_core_params_in_model_name(self.parameter_dictionary) + \
                      CentralisedGALearner.get_additional_params_in_model_name(self.parameter_dictionary) + \
                      [initial_fitness, genome_fitness, model_name]
        results = ",".join([str(element) for element in result_data])

        results_file.write(f"{results}\n")
        results_file.close()

    def generate_model_name(self, fitness):
        return CentralisedGALearner.get_model_name_from_dictionary(self.parameter_dictionary, fitness)

    @staticmethod
    def get_model_name_from_dictionary(parameter_dictionary, fitness):
        """
        Create a name string for a model generated using the given parameter file, its rank and fitness value

        @param parameter_dictionary:  Dictionary containing all parameters for the experiment
        @param fitness: Fitness of the model to be saved
        @return: The model name as a string
        """

        parameters_in_name = CentralisedLearner.get_core_params_in_model_name(parameter_dictionary)
        parameters_in_name += GALearner.get_additional_params_in_model_name(parameter_dictionary)

        # Get fitness
        parameters_in_name += [fitness]

        # Put all in a string and return
        return "_".join([str(param) for param in parameters_in_name])
