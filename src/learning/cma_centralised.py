import json

import cma
import copy
import sys
import os
import numpy as np
import ray

from fitness import FitnessCalculator
from learning.learner_centralised import CentralisedLearner
from learning.cma_parent import CMALearner
from glob import glob
from io import StringIO
from operator import add
from learning.rwg_centralised import CentralisedRWGLearner
from helpers import novelty_helpers
from functools import partial
from collections import deque


@ray.remote
def learn_in_parallel(fitness_calculator, agent_pop, spec_flag):
    return fitness_calculator.calculate_fitness_of_agent_population(agent_pop, spec_flag)


class CentralisedCMALearner(CentralisedLearner, CMALearner):
    def __init__(self, calculator):
        super().__init__(calculator)
        self.calculate_behaviour_distance = partial(novelty_helpers.calculate_distance, metric=self.novelty_params['distance_metric'])


    def learn(self, logging=True):
        """
        Search for the best genome that solves the problem using CMA-ES, while also saving the models every so often
        and logging the results to a result file for analysis.

        @return: The best genome found by CMA-ES and its fitness
        """

        # Put CMA output in a buffer for logging to a file at the end of the function
        old_stdout = sys.stdout
        sys.stdout = mystdout = StringIO()

        options = {'seed': self.parameter_dictionary['general']['seed'],
                   'maxiter': self.parameter_dictionary['algorithm']['cma']['generations'],
                   'popsize': self.get_genome_population_length(),
                   'tolx': self.parameter_dictionary['algorithm']['cma']['tolx'],
                   'tolfunhist': self.parameter_dictionary['algorithm']['cma']['tolfunhist'],
                   'tolflatfitness': self.parameter_dictionary['algorithm']['cma']['tolflatfitness'],
                   'tolfun': self.parameter_dictionary['algorithm']['cma']['tolfun'],
                   'tolstagnation': self.parameter_dictionary['algorithm']['cma']['tolstagnation'],
                   'tolfacupx': self.parameter_dictionary['algorithm']['cma']['tolfacupx']}

        # Initialise cma with a mean genome and sigma
        seed_genome, seed_fitness = self.get_seed_genome()
        es = cma.CMAEvolutionStrategy(seed_genome, self.parameter_dictionary['algorithm']['cma']['sigma'], options)
        num_threads = self.num_agents

        if self.parameter_dictionary["algorithm"]["agent_population_size"] % num_threads != 0:
            raise RuntimeError("Agent population is not divisible by the number of parallel threads")

        best_genome = None
        best_genome_fitness = float('-inf')
        generation = 0

        if self.using_novelty:
            recent_insertions = deque(maxlen=self.novelty_params['insertions_before_increase'])
            np_random = np.random.RandomState(self.parameter_dictionary['general']['seed'])

        # Learning loop
        while not es.stop():
            # Get population of genomes to be used this generation
            genome_population = es.ask()

            # For homogeneous teams rewarding at the individual level,
            if self.team_type == "homogeneous" and self.reward_level == "individual":
                """
                                new_population = []

                                for ind in genome_population:
                                    ind_1 = copy.deepcopy(ind)
                                    ind_2 = copy.deepcopy(ind)
                                    new_population += [ind_1, ind_2]

                                genome_population = new_population
                """
                raise RuntimeError("This configuration is not supported yet")

            # Convert genomes to agents
            controller_population = self.convert_genomes_to_controllers(genome_population)

            if self.learning_type == "centralised":
                agent_population = controller_population
                agent_pop_size = len(agent_population)

            agent_fitness_lists = []

            if self.multithreading:
                if self.learning_type == "fully-centralised":
                    raise RuntimeError("Multithreading not supported yet for fully centralised learning")

                else:
                    remainder_agents = agent_pop_size % (self.num_agents**2)
                    divisible_pop_size = agent_pop_size - remainder_agents
                    parallel_threads = []

                    for i in range(0, num_threads+1):
                        start = i * (divisible_pop_size//num_threads)
                        end = (i+1) * (divisible_pop_size//num_threads)
                        mini_pop = agent_population[start:end]

                        # Makes sure that each core gets a population that can be divided into teams of size self.num_agents
                        if i == num_threads and remainder_agents != 0:
                            mini_pop += agent_population[end:end+remainder_agents]

                        parallel_threads += [learn_in_parallel.remote(self.fitness_calculator, mini_pop, self.calculate_specialisation)]

                    parallel_results = ray.get(parallel_threads)
                    #team_specialisations = []

                    for element in parallel_results:
                        agent_fitness_lists += element[0]
                        #team_specialisations += element[1]

            else:
                agent_fitness_lists, team_specialisations, agent_bc_vectors = self.fitness_calculator.calculate_fitness_of_agent_population(controller_population, self.calculate_specialisation)

            # Convert agent fitnesses into genome fitnesses
            genome_fitness_lists = self.get_genome_fitnesses_from_agent_fitnesses(agent_fitness_lists)
            genome_fitness_average = [np.mean(fitness_list) for fitness_list in genome_fitness_lists]
            team_bc_vectors = self.get_team_bc_from_agent_bc(agent_bc_vectors)

            # Update the algorithm with the new fitness evaluations
            # CMA minimises fitness so we negate the fitness values
            if not self.using_novelty:
                es.tell(genome_population, [-f for f in genome_fitness_average])

            else:
                novelties = [0] * len(team_bc_vectors)
                min_fitness = float('inf')
                max_fitness = float('-inf')
                max_fitness_genome = None
                min_novelty = float('inf')
                max_novelty = float('-inf')

                # Get each genome's novelty
                for index, bc in enumerate(team_bc_vectors):
                    # Calculate distance to each behaviour in the archive.
                    distances = {}

                    if len(self.novelty_archive) > 0:
                        for key in self.novelty_archive.keys():
                            distances[str(key)] = self.calculate_behaviour_distance(a=bc, b=self.novelty_archive[key]['bc'])

                    # Calculate distances to the current population if the archive is empty
                    else:
                        for other_index,genome in enumerate(genome_population):
                            if other_index != index:
                                distances[str(genome)] = self.calculate_behaviour_distance(a=bc, b=team_bc_vectors[other_index])

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

                        else:
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
                novelty_range = max_novelty-min_novelty
                fitness_range = max_fitness-min_fitness

                for index in range(len(genome_fitness_average)):
                    novelties[index] = (novelties[index] - min_novelty) / novelty_range
                    genome_fitness_average[index] = (genome_fitness_average[index] - min_fitness) / fitness_range

                # Store best fitness if it's better than the best so far
                if max_fitness > best_genome_fitness:
                    best_genome_fitness = max_fitness
                    best_genome = list(max_fitness_genome)

                # Calculate weighted score of each solution and add to archive if it passes the threshold
                weighted_scores = [0] * len(genome_fitness_average)

                for index in range(len(genome_fitness_average)):
                    weighted_scores[index] = (1 - self.novelty_params['novelty_weight']) * genome_fitness_average[index] + self.novelty_params['novelty_weight'] * novelties[index]

                    if weighted_scores[index] > self.novelty_params['archive_threshold']:
                        key = str(genome_population[index])
                        self.novelty_archive[key] = {'bc': team_bc_vectors[index],
                                                     'fitness': genome_fitness_average[index]}
                        recent_insertions.append(generation)

                    else:
                        # Insert genome into archive with certain probability, regardless of threshold
                        if np_random.random() < self.novelty_params['random_insertion_chance']:
                            recent_insertions.append(generation)

                # Update population distribution
                es.tell(genome_population, [-s for s in weighted_scores])

                # Increase archive threshold if too many things have been recently inserted
                if len(recent_insertions) >= self.novelty_params['insertions_before_increase']:
                    insertion_window_start = recent_insertions.popleft()

                    if (generation - insertion_window_start) <= self.novelty_params['gens_before_change']:
                        increased_threshold = self.novelty_params['archive_threshold'] + self.novelty_params['threshold_increase_amount']
                        self.novelty_params['archive_threshold'] = min(1.0, increased_threshold)

                # Decrease archive threshold if not enough things have been recently inserted
                last_insertion_time = recent_insertions.pop()
                recent_insertions.append(last_insertion_time)

                if (generation - last_insertion_time) >= self.novelty_params['gens_before_change']:
                    decreased_threshold = self.novelty_params['archive_threshold'] - self.novelty_params['threshold_decrease_amount']
                    self.novelty_params['archive_threshold'] = max(0.0, decreased_threshold)

            generation += 1

            if generation % self.logging_rate == 0:
                self.log(best_genome, best_genome_fitness, generation, seed_fitness)

            es.disp()

        print(f"Best fitness is {best_genome_fitness}")

        print(es.stop())

        # Save best model
        model_name = self.generate_model_name(best_genome_fitness)
        self.log(best_genome, best_genome_fitness, "final", seed_fitness)

        # Log evolution details to file
        log_file_name = model_name + ".log"
        f = open(log_file_name, "w")
        f.write(mystdout.getvalue())
        f.close()

        # Reset output stream
        sys.stdout = old_stdout

        return best_genome, best_genome_fitness

    # Helpers ---------------------------------------------------------------------------------------------------------

    def get_genome_population_length(self):
        """
        Calculates how many genomes should be in the CMA population based on the number of agents

        @return: Integer representing the length of the genome population
        """
        # If teams are heterogneous and rewards are individual, there must be one genome per agent
        if self.team_type == "heterogeneous" and self.reward_level == "individual":
            return self.parameter_dictionary['algorithm']['agent_population_size']

        # For most configurations, each genome is either copied onto multiple agents or split across multiple agents
        else:
            return self.parameter_dictionary['algorithm']['agent_population_size'] / self.num_agents

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

    def log(self, genome, genome_fitness, generation, seed_fitness):
        """
        Save the genome model and save fitness and parameters to a results file

        @param genome: The genome being logged
        @param genome_fitness: The fitness of the genome
        @param generation: The current generation of CMA-ES
        @param seed_fitness: The fitness of the seed genome
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
            result_headings += ["seed_fitness", "fitness", "model_name"]
            result_headings = ",".join(result_headings)
            result_headings += "\n"
            results_file.write(result_headings)

        else:
            results_file = open(results_filename, 'a')

        result_data = CentralisedLearner.get_core_params_in_model_name(self.parameter_dictionary) + \
                      CentralisedCMALearner.get_additional_params_in_model_name(self.parameter_dictionary) + \
                      [seed_fitness, genome_fitness, model_name]
        results = ",".join([str(element) for element in result_data])

        results_file.write(f"{results}\n")
        results_file.close()

    def generate_model_name(self, fitness):
        return CentralisedCMALearner.get_model_name_from_dictionary(self.parameter_dictionary, fitness)

    @staticmethod
    def get_model_name_from_dictionary(parameter_dictionary, fitness):
        """
        Create a name string for a model generated using the given parameter file, its rank and fitness value

        @param parameter_dictionary:  Dictionary containing all parameters for the experiment
        @param fitness: Fitness of the model to be saved
        @return: The model name as a string
        """

        parameters_in_name = CentralisedLearner.get_core_params_in_model_name(parameter_dictionary)
        parameters_in_name += CMALearner.get_additional_params_in_model_name(parameter_dictionary)

        # Get fitness
        parameters_in_name += [fitness]

        # Put all in a string and return
        return "_".join([str(param) for param in parameters_in_name])

