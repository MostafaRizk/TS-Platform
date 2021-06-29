import json

import cma
import copy
import sys
import os
import numpy as np
import ray

from fitness import FitnessCalculator
from learning.learner_decentralised import DecentralisedLearner
from learning.cma_parent import CMALearner
from glob import glob
from io import StringIO
from collections import deque

from learning.rwg_centralised import CentralisedRWGLearner


def learn_agent(learner, learner_index, fitness_calculator, insert_representative_genomes_in_population, remove_representative_fitnesses, convert_genomes_to_agents, calculate_specialisation, using_novelty, remove_representative_bc, novelty_archive, novelty_params, calculate_behaviour_distance, recent_insertions, np_random, generation):
    new_learner = copy.deepcopy(learner)

    # Get population of genomes to be used this generation
    genome_population = new_learner.ask()

    # Convert genomes to agents
    extended_genome_population = insert_representative_genomes_in_population(genome_population, learner_index)
    agent_population = convert_genomes_to_agents(extended_genome_population)

    # Get fitnesses of genomes (same as fitnesses of agents)
    genome_fitness_lists, team_specialisations, agent_bc_vectors = fitness_calculator.calculate_fitness_of_agent_population(agent_population, calculate_specialisation)

    # Remove fitness values of the representative agents of the other populations
    genome_fitness_lists = remove_representative_fitnesses(genome_fitness_lists)
    genome_fitness_average = [np.mean(fitness_list) for fitness_list in genome_fitness_lists]
    agent_bc_vectors = remove_representative_bc(agent_bc_vectors)

    best_genome = None
    best_genome_fitness = float('-inf')

    # Update the algorithm with the new fitness evaluations
    # CMA minimises fitness so we negate the fitness values
    if not using_novelty:
        new_learner.tell(genome_population, [-f for f in genome_fitness_average])

    else:
        novelties = [0] * len(agent_bc_vectors)
        min_fitness = float('inf')
        max_fitness = float('-inf')
        max_fitness_genome = None
        min_novelty = float('inf')
        max_novelty = float('-inf')

        # Get each genome's novelty
        for index, bc in enumerate(agent_bc_vectors):
            # Calculate distance to each behaviour in the archive.
            distances = {}

            if len(novelty_archive[learner_index]) > 0:
                for key in novelty_archive[learner_index].keys():
                    distances[str(key)] = calculate_behaviour_distance(a=bc, b=novelty_archive[learner_index][key]['bc'])

            # Calculate distances to the current population if the archive is empty
            else:
                for other_index, genome in enumerate(genome_population):
                    if other_index != index:
                        distances[str(genome)] = calculate_behaviour_distance(a=bc, b=agent_bc_vectors[other_index])

            # Find k-nearest-neighbours in the archive
            # TODO: Make this more efficient
            nearest_neighbours = {}

            for i in range(novelty_params['k']):
                min_dist = float('inf')
                min_key = None

                if len(novelty_archive[learner_index]) > 0:
                    for key in novelty_archive[learner_index].keys():
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

            avg_distance /= novelty_params['k']
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
            novelties[index] = (novelties[index] - min_novelty) / novelty_range
            genome_fitness_average[index] = (genome_fitness_average[index] - min_fitness) / fitness_range

        # Store best fitness if it's better than the best so far
        if max_fitness > best_genome_fitness:
            best_genome_fitness = max_fitness
            best_genome = list(max_fitness_genome)

        # Calculate weighted score of each solution and add to archive if it passes the threshold
        weighted_scores = [0] * len(genome_fitness_average)

        for index in range(len(genome_fitness_average)):
            weighted_scores[index] = (1 - novelty_params['novelty_weight']) * genome_fitness_average[index] + \
                                     novelty_params['novelty_weight'] * novelties[index]

            if weighted_scores[index] > novelty_params['archive_threshold'][learner_index]:
                key = str(genome_population[index])
                novelty_archive[learner_index][key] = {'bc': agent_bc_vectors[index],
                                                       'fitness': genome_fitness_average[index]}
                recent_insertions[learner_index].append(generation)

            else:
                # Insert genome into archive with certain probability, regardless of threshold
                if np_random[learner_index].random() < novelty_params['random_insertion_chance']:
                    recent_insertions[learner_index].append(generation)

        # Update population distribution
        new_learner.tell(genome_population, [-s for s in weighted_scores])

        # Increase archive threshold if too many things have been recently inserted
        if len(recent_insertions[learner_index]) >= novelty_params['insertions_before_increase']:
            insertion_window_start = recent_insertions[learner_index].popleft()

            if (generation - insertion_window_start) <= novelty_params['gens_before_change']:
                increased_threshold = novelty_params['archive_threshold'][learner_index] + novelty_params['threshold_increase_amount']
                novelty_params['archive_threshold'][learner_index] = min(1.0, increased_threshold)

        # Decrease archive threshold if not enough things have been recently inserted
        if len(recent_insertions[learner_index]) > 0:
            last_insertion_time = recent_insertions[learner_index].pop()
            recent_insertions[learner_index].append(last_insertion_time)

        else:
            last_insertion_time = -1

        if (generation - last_insertion_time) >= novelty_params['gens_before_change']:
            decreased_threshold = novelty_params['archive_threshold'][learner_index] - novelty_params['threshold_decrease_amount']
            novelty_params['archive_threshold'][learner_index] = max(0.0, decreased_threshold)

    # Update representative agent
    #best_genome = new_learner.result[0]
    #best_fitness = -new_learner.result[1]

    return new_learner, best_genome, best_genome_fitness


@ray.remote
def learn_in_parallel(learner, index, fitness_calculator, insert_representative_genomes_in_population, remove_representative_fitnesses, convert_genomes_to_agents, calculate_specialisation, using_novelty, remove_representative_bc, novelty_archive, novelty_params, calculate_behaviour_distance, recent_insertions, np_random, generation):
    return learn_agent(learner, index, fitness_calculator, insert_representative_genomes_in_population, remove_representative_fitnesses, convert_genomes_to_agents, calculate_specialisation, using_novelty, remove_representative_bc, novelty_archive, novelty_params, calculate_behaviour_distance, recent_insertions, np_random, generation)


@ray.remote
def empty_task():
    return None


class DecentralisedCMALearner(DecentralisedLearner, CMALearner):
    def __init__(self, calculator):
        super().__init__(calculator)

    def learn(self, logging=True):
        """
        Search for the best team of genomes that solve the problem using CMA-ES, while also saving the models every so often
        and logging the results to a result file for analysis.

        @return: The best genome found by CMA-ES and its fitness
        """

        # Put CMA output in a buffer for logging to a file at the end of the function
        old_stdout = sys.stdout
        sys.stdout = mystdout = StringIO()

        options = {'seed': self.parameter_dictionary['general']['seed'],
                   'maxiter': self.parameter_dictionary['algorithm']['cma']['generations'],
                   'popsize': self.parameter_dictionary['algorithm']['agent_population_size']//self.num_agents,
                   'tolx': self.parameter_dictionary['algorithm']['cma']['tolx'],
                   'tolfunhist': self.parameter_dictionary['algorithm']['cma']['tolfunhist'],
                   'tolflatfitness': self.parameter_dictionary['algorithm']['cma']['tolflatfitness'],
                   'tolfun': self.parameter_dictionary['algorithm']['cma']['tolfun'],
                   'tolstagnation': self.parameter_dictionary['algorithm']['cma']['tolstagnation'],
                   'tolfacupx': self.parameter_dictionary['algorithm']['cma']['tolfacupx']}

        # Get seed genome
        seed_genome, seed_fitness = self.get_seed_genome()  # TODO: Investigate whether learners should start from different seed genomes
        learners = []
        stopping_reasons = []
        best_fitnesses = []

        if self.using_novelty:
            recent_insertions = [deque(maxlen=self.novelty_params['insertions_before_increase']) for _ in range(self.num_agents)]
            np_random = [np.random.RandomState(self.parameter_dictionary['general']['seed']) for _ in range(self.num_agents)]

        # Initialise each cma instance and representative agents
        for i in range(self.num_agents):
            learners += [cma.CMAEvolutionStrategy(seed_genome, self.parameter_dictionary['algorithm']['cma']['sigma'], options)]
            self.representative_genomes += [seed_genome]
            stopping_reasons += [None]
            best_fitnesses += [None]

        for generation in range(self.parameter_dictionary['algorithm']['cma']['generations']):

            if self.multithreading:
                parallel_threads = []

                # Create and run parallel threads
                for index, learner in enumerate(learners):
                    if not learner.stop():
                        parallel_threads += [learn_in_parallel.remote(learner, index, self.fitness_calculator, self.insert_representative_genomes_in_population, self.remove_representative_fitnesses, self.convert_genomes_to_controllers, self.calculate_specialisation, self.using_novelty, self.remove_representative_bc, self.novelty_archive, self.novelty_params, self.calculate_behaviour_distance, recent_insertions, np_random, generation)]

                    elif not stopping_reasons[index]:
                        # Log reason for stopping
                        parallel_threads += [empty_task.remote()]
                        stopping_reasons[index] = [learner.stop()]
                        print(stopping_reasons[index])

                parallel_results = ray.get(parallel_threads)

                # Update shared data and log according to results
                for index in range(len(learners)):
                    if parallel_results[index]:
                        updated_learner, best_genome, best_fitness = parallel_results[index]

                        learners[index] = updated_learner
                        best_fitnesses[index] = best_fitness
                        self.representative_genomes[index] = best_genome

                        # Log best genome for this learner
                        if generation % self.logging_rate == 0:
                            self.log(best_genome, best_fitness, generation, seed_fitness, index)

                        updated_learner.disp()

            else:
                for index, learner in enumerate(learners):
                    if not learner.stop():
                        learners[index], self.representative_genomes[index], best_fitnesses[index] = learn_agent(learner, index, self.fitness_calculator, self.insert_representative_genomes_in_population, self.remove_representative_fitnesses, self.convert_genomes_to_controllers, self.calculate_specialisation, self.using_novelty, self.remove_representative_bc, self.novelty_archive, self.novelty_params, self.calculate_behaviour_distance, recent_insertions, np_random, generation)

                        # Log best genome for this learner
                        if generation % self.logging_rate == 0:
                            self.log(self.representative_genomes[index], best_fitnesses[index], generation, seed_fitness, index)

                        learner.disp()

                    elif not stopping_reasons[index]:
                        # Log reason for stopping
                        stopping_reasons[index] = [learner.stop()]
                        print(stopping_reasons[index])

        # Print fitness of best representative from each population
        print(f"Best fitnesses are {best_fitnesses}")

        # Save best model for each representative agent
        for i in range(len(best_fitnesses)):
            model_name = self.generate_model_name(best_fitnesses[i], i)
            self.log(self.representative_genomes[i], best_fitnesses[i], "final", seed_fitness, i)

        # Log evolution details to file
        # TODO: Modify this to have separate log for each learner?
        team_fitness = sum(best_fitnesses)
        model_name = self.generate_model_name(team_fitness, 'all')
        log_file_name = model_name + ".log"
        f = open(log_file_name, "w")
        f.write(mystdout.getvalue())
        f.close()

        # Reset output stream
        sys.stdout = old_stdout

        # Return best genome for each population and its fitness
        return self.representative_genomes, best_fitnesses

    # Helpers ---------------------------------------------------------------------------------------------------------

    def log(self, genome, genome_fitness, generation, seed_fitness, agent_index):
        """
        Save the genome model and save fitness and parameters to a results file

        @param genome: The genome being logged
        @param genome_fitness: The fitness of the genome
        @param generation: The current generation of CMA-ES
        @param seed_fitness: The fitness of the seed genome
        @param agent_index: The index of the agent on the team
        @return:
        """
        # Save model
        model_name = self.generate_model_name(genome_fitness, agent_index)
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

        result_data = DecentralisedLearner.get_core_params_in_model_name(self.parameter_dictionary) + \
                      DecentralisedCMALearner.get_additional_params_in_model_name(self.parameter_dictionary) + \
                      [seed_fitness, genome_fitness, model_name]
        results = ",".join([str(element) for element in result_data])

        results_file.write(f"{results}\n")
        results_file.close()

    def generate_model_name(self, fitness, agent_index):
        return DecentralisedCMALearner.get_model_name_from_dictionary(self.parameter_dictionary, fitness, agent_index)

    @staticmethod
    def get_model_name_from_dictionary(parameter_dictionary, fitness, agent_index):
        """
        Create a name string for a model generated using the given parameter file, its rank and fitness value

        @param parameter_dictionary: Dictionary containing all parameters for the experiment
        @param fitness: Fitness of the model to be saved
        @param agent_index: Index of the given agent on the team
        @return: The model name as a string
        """

        parameters_in_name = DecentralisedLearner.get_core_params_in_model_name(parameter_dictionary)
        parameters_in_name += CMALearner.get_additional_params_in_model_name(parameter_dictionary)

        parameters_in_name += [agent_index]

        # Get fitness
        parameters_in_name += [fitness]

        # Put all in a string and return
        return "_".join([str(param) for param in parameters_in_name])