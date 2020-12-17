import json

import cma
import copy
import sys
import os
import numpy as np

from fitness import FitnessCalculator
from learning.learner_decentralised import DecentralisedLearner
from learning.cma_parent import CMALearner
from glob import glob
from io import StringIO

from learning.rwg import RWGLearner


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
                   'popsize': self.parameter_dictionary['algorithm']['agent_population_size'],
                   'tolx': self.parameter_dictionary['algorithm']['cma']['tolx'],
                   'tolfunhist': self.parameter_dictionary['algorithm']['cma']['tolfunhist'],
                   'tolflatfitness': self.parameter_dictionary['algorithm']['cma']['tolflatfitness'],
                   'tolfun': self.parameter_dictionary['algorithm']['cma']['tolfun']}

        # Get seed genome
        seed_genome, seed_fitness = self.get_seed_genome()  # TODO: Investigate whether learners should start from different seed genomes
        learners = []
        stopping_reasons = []
        best_fitnesses = []

        # Initialise each cma instance and representative agents
        for i in range(self.num_agents):
            learners += [cma.CMAEvolutionStrategy(seed_genome, self.parameter_dictionary['algorithm']['cma']['sigma'], options)]
            self.representative_genomes += [seed_genome]
            stopping_reasons += [None]
            best_fitnesses += [None]

        for generation in range(self.parameter_dictionary['algorithm']['cma']['generations']):
            for index, learner in enumerate(learners):
                if not learner.stop():
                    # Get population of genomes to be used this generation
                    genome_population = learner.ask()

                    # Convert genomes to agents
                    extended_genome_population = self.insert_representative_genomes_in_population(genome_population, index)
                    agent_population = self.convert_genomes_to_agents(extended_genome_population)

                    # Get fitnesses of genomes (same as fitnesses of agents)
                    genome_fitness_lists = self.fitness_calculator.calculate_fitness_of_agent_population(agent_population)

                    # Remove fitness values of the representative agents of the other populations
                    genome_fitness_lists = self.remove_representative_fitnesses(genome_fitness_lists)
                    genome_fitness_average = [np.mean(fitness_list) for fitness_list in genome_fitness_lists]

                    # Update the algorithm with the new fitness evaluations
                    # CMA minimises fitness so we negate the fitness values
                    learner.tell(genome_population, [-f for f in genome_fitness_average])

                    # Update representative agent
                    best_genome = learner.result[0]
                    best_fitness = -learner.result[1]
                    self.representative_genomes[index] = best_genome
                    best_fitnesses[index] = best_fitness

                    # Log best genome for this learner
                    if generation % self.logging_rate == 0:
                        self.log(best_genome, best_fitness, generation, seed_fitness, index)

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
        # TODO: Modify this to have separate log for each learner
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