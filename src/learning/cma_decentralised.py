import json

import cma
import copy
import sys
import os
import numpy as np

from fitness import FitnessCalculator
from learning.learner_decentralised import DecentralisedLearner
from glob import glob
from io import StringIO

from learning.rwg import RWGLearner


class DecentralisedCMALearner(DecentralisedLearner):
    def __init__(self, calculator):
        super().__init__(calculator)

        if self.parameter_dictionary['general']['algorithm_selected'] != "cma" and \
                self.parameter_dictionary['general']['algorithm_selected'] != "cma_with_seeding":
            raise RuntimeError(f"Cannot run cma. Parameters request "
                               f"{self.parameter_dictionary['general']['algorithm_selected']}")

        # Log every x many generations
        self.logging_rate = self.parameter_dictionary['algorithm']['cma']['logging_rate']

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
        seed_genome, seed_fitness = self.get_seed_genome() # TODO: Investigate whether learners should start from different seed genomes
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
                    # TODO: Implement representative insertion method
                    self.insert_representative_genomes_in_population(genome_population, self.representative_genomes)
                    agent_population = self.convert_genomes_to_agents(genome_population)

                    # Get fitnesses of genomes (same as fitnesses of agents)
                    genome_fitness_lists = self.fitness_calculator.calculate_fitness_of_agent_population(agent_population)

                    # Remove fitness values of the representative agents of the other populations
                    # TODO: Implement representative removal method
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
                        # TODO: Modify logging to log agent index
                        self.log(best_genome, best_fitness, generation, seed_fitness)

                    learner.disp()

                elif not stopping_reasons[index]:
                    # Log reason for stopping
                    stopping_reasons[index] = [learner.stop()]
                    print(stopping_reasons[index])

        # Print fitness of best representative from each population
        print(f"Best fitnesses are {best_fitnesses}")

        # Save best model for each representative agent
        # TODO: Save best model for each learner
        model_name = self.generate_model_name(best_fitness)
        self.log(best_genome, best_fitness, "final", seed_fitness)

        # Log evolution details to file
        log_file_name = model_name + ".log"
        f = open(log_file_name, "w")
        f.write(mystdout.getvalue())
        f.close()

        # Reset output stream
        sys.stdout = old_stdout

        # Return best genome for each population and its fitness
        return best_genome, best_fitness # TODO: Return best team