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

        if self.using_novelty:
            self.recent_insertions = deque(maxlen=self.novelty_params['insertions_before_increase'])

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
            scores, best_genome, best_genome_fitness = self.calculate_weighted_scores(genome_population, genome_fitness_average, team_bc_vectors, 0)
            initial_fitness = best_genome_fitness

            for ind, score in zip(genome_population, scores):
                ind.fitness.values = (score,)

            for generation in range(1, num_generations+1):
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
                    genome_fitness_lists[loser_index] = child_genome_fitness_lists[i]
                    genome_fitness_average[loser_index] = child_genome_fitness_average[i]
                    team_bc_vectors[loser_index] = child_team_bc_vectors[i]

                # Calculate novelty
                weighted_scores, max_fitness_genome, max_fitness = self.calculate_weighted_scores(genome_population, genome_fitness_average, team_bc_vectors, generation)

                for ind, score in zip(genome_population, weighted_scores):
                    ind.fitness.values = (score,)

                if max_fitness > best_genome_fitness:
                    best_genome_fitness = max_fitness
                    best_genome = max_fitness_genome

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

    def calculate_weighted_scores(self, genome_population_ref, genome_fitness_average_ref, genome_bc_vectors_ref, generation):

        genome_population = genome_population_ref[:]
        genome_fitness_average = genome_fitness_average_ref[:]
        genome_bc_vectors = genome_bc_vectors_ref[:]

        novelties = [0] * len(genome_bc_vectors)
        min_fitness = float('inf')
        max_fitness = float('-inf')
        max_fitness_genome = None
        min_novelty = float('inf')
        max_novelty = float('-inf')

        # Get each genome's novelty
        for index, bc in enumerate(genome_bc_vectors):
            # Calculate distance to each behaviour in the archive.
            distances = {}

            if len(self.novelty_archive) > 0:
                for key in self.novelty_archive.keys():
                    distances[str(key)] = self.calculate_behaviour_distance(a=bc, b=self.novelty_archive[key]['bc'])

            # Calculate distances to the current population
            for other_index, genome in enumerate(genome_population):
                if other_index != index:
                    distances[str(genome)] = self.calculate_behaviour_distance(a=bc, b=genome_bc_vectors[other_index])

            # Find k-nearest-neighbours in the archive
            # TODO: Make this more efficient
            nearest_neighbours = {}

            for i in range(self.novelty_params['k']):
                min_dist = float('inf')
                min_key = None

                if len(self.novelty_archive) > 0:
                    for key in self.novelty_archive.keys():
                        if distances[key] < min_dist and key not in nearest_neighbours:
                            min_dist = distances[key]
                            min_key = key

                for other_index, genome in enumerate(genome_population):
                    key = str(genome)
                    if other_index != index:
                        if distances[key] < min_dist and key not in nearest_neighbours:
                            min_dist = distances[key]
                            min_key = key

                if min_key:
                    nearest_neighbours[min_key] = min_key

            # Calculate novelty (average distance to k nearest neighbours)
            avg_distance = 0

            for key in nearest_neighbours.keys():
                avg_distance += distances[key]

            avg_distance /= self.novelty_params['k']
            novelties[index] = avg_distance

            if novelties[index] < min_novelty:
                min_novelty = novelties[index]

            if novelties[index] > max_novelty:
                max_novelty = novelties[index]

            if genome_fitness_average[index] < min_fitness:
                min_fitness = genome_fitness_average[index]

            if genome_fitness_average[index] > max_fitness:
                max_fitness = genome_fitness_average[index]
                max_fitness_genome = genome_population[index]

        # Normalise novelties and fitnesses
        novelty_range = max_novelty - min_novelty
        fitness_range = max_fitness - min_fitness

        for index in range(len(genome_fitness_average)):
            if novelty_range != 0:
                 novelties[index] = (novelties[index] - min_novelty) / novelty_range

            if fitness_range != 0:
                genome_fitness_average[index] = (genome_fitness_average[index] - min_fitness) / fitness_range

        # Calculate weighted score of each solution and add to archive if it passes the threshold
        weighted_scores = [0] * len(genome_fitness_average)

        for index in range(len(genome_fitness_average)):
            weighted_scores[index] = (1 - self.novelty_params['novelty_weight']) * genome_fitness_average[index] + self.novelty_params['novelty_weight'] * novelties[index]

            if weighted_scores[index] > self.novelty_params['archive_threshold']:
                key = str(genome_population[index])
                self.novelty_archive[key] = {'bc': genome_bc_vectors[index], 'fitness': genome_fitness_average[index]}
                self.recent_insertions.append(generation)

            else:
                # Insert genome into archive with certain probability, regardless of threshold
                if self.np_random.random() < self.novelty_params['random_insertion_chance']:
                    self.recent_insertions.append(generation)

            # Increase archive threshold if too many things have been recently inserted
            if len(self.recent_insertions) >= self.novelty_params['insertions_before_increase']:
                insertion_window_start = self.recent_insertions.popleft()

                if (generation - insertion_window_start) <= self.novelty_params['gens_before_change']:
                    increased_threshold = self.novelty_params['archive_threshold'] + self.novelty_params['threshold_increase_amount']
                    self.novelty_params['archive_threshold'] = min(1.0, increased_threshold)

            # Decrease archive threshold if not enough things have been recently inserted
            if len(self.recent_insertions) > 0:
                last_insertion_time = self.recent_insertions.pop()
                self.recent_insertions.append(last_insertion_time)

            else:
                last_insertion_time = -1

            if (generation - last_insertion_time) >= self.novelty_params['gens_before_change']:
                decreased_threshold = self.novelty_params['archive_threshold'] - self.novelty_params['threshold_decrease_amount']
                self.novelty_params['archive_threshold'] = max(0.0, decreased_threshold)

        return weighted_scores, list(max_fitness_genome), max_fitness

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
