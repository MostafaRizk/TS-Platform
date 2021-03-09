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

from learning.rwg import RWGLearner


@ray.remote
def learn_in_parallel(fitness_calculator, agent_pop, spec_flag):
    return fitness_calculator.calculate_fitness_of_agent_population(agent_pop, spec_flag)


class CentralisedCMALearner(CentralisedLearner, CMALearner):
    def __init__(self, calculator):
        super().__init__(calculator)

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
                   'tolfun': self.parameter_dictionary['algorithm']['cma']['tolfun']}

        # Initialise cma with a mean genome and sigma
        seed_genome, seed_fitness = self.get_seed_genome()
        es = cma.CMAEvolutionStrategy(seed_genome, self.parameter_dictionary['algorithm']['cma']['sigma'], options)
        num_threads = self.num_agents
        ray.init(address=os.environ["ip_head"])

        if self.parameter_dictionary["algorithm"]["agent_population_size"] % num_threads != 0:
            raise RuntimeError("Agent population is not divisible by the number of parallel threads")

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
            agent_population = self.convert_genomes_to_agents(genome_population)
            agent_pop_size = len(agent_population)
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
            agent_fitness_lists = []
            #team_specialisations = []

            for element in parallel_results:
                agent_fitness_lists += element[0]
                #team_specialisations += element[1]

            # Convert agent fitnesses into genome fitnesses
            genome_fitness_lists = self.get_genome_fitnesses_from_agent_fitnesses(agent_fitness_lists)
            genome_fitness_average = [np.mean(fitness_list) for fitness_list in genome_fitness_lists]

            # Update the algorithm with the new fitness evaluations
            # CMA minimises fitness so we negate the fitness values
            es.tell(genome_population, [-f for f in genome_fitness_average])

            generation = es.result.iterations

            if generation % self.logging_rate == 0:
                self.log(es.result[0], -es.result[1], generation, seed_fitness)

            es.disp()

        # Get best genome and its fitness value
        best_genome = es.result[0]
        best_fitness = -es.result[1]

        print(f"Best fitness is {best_fitness}")

        print(es.stop())

        # Save best model
        model_name = self.generate_model_name(best_fitness)
        self.log(best_genome, best_fitness, "final", seed_fitness)

        # Log evolution details to file
        log_file_name = model_name + ".log"
        f = open(log_file_name, "w")
        f.write(mystdout.getvalue())
        f.close()

        # Reset output stream
        sys.stdout = old_stdout

        return best_genome, best_fitness

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

