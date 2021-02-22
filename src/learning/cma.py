import json

import cma
import copy
import sys
import os
import numpy as np
import signal

from fitness import FitnessCalculator
from learning.learner_parent import Learner
from glob import glob
from io import StringIO

from learning.rwg import RWGLearner


class CMALearner(Learner):
    def __init__(self, calculator):
        super().__init__(calculator)

        if self.parameter_dictionary['general']['algorithm_selected'] != "cma" and \
                self.parameter_dictionary['general']['algorithm_selected'] != "cma_with_seeding" and \
                self.parameter_dictionary['general']['algorithm_selected'] != "partialcma":
            raise RuntimeError(f"Cannot run cma. Parameters request "
                               f"{self.parameter_dictionary['general']['algorithm_selected']}")

        # Log every x many generations
        self.logging_rate = self.parameter_dictionary['algorithm']['cma']['logging_rate']

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

        # Learning loop
        while not es.stop():

            # Graceful exit (allows to pick up from where the algorithm last left off)
            # Not yet tested
            def signal_handler(*args):
                # Create crash folder (if it doesn't exist)
                dir_name = "../crashed_runs"
                if not os.path.isdir(dir_name):
                    os.mkdir(dir_name)

                # Calculate latest sigma
                sigma = es.sigma0

                # Calculate remaining generations
                remaining_generations = options['maxiter'] - es.result.iterations

                # Save parameter dictionary as json with relevant modified parameters
                new_dictionary = copy.deepcopy(self.parameter_dictionary)
                new_dictionary["general"]["algorithm_selected"] = "partialcma"
                new_dictionary['algorithm']['cma']['sigma'] = sigma
                new_dictionary['algorithm']['cma']['generations'] = remaining_generations
                new_dictionary['algorithm']['cma']['seeding_required'] = True  # So that it looks for a seed genome

                parameters_in_filename = []
                parameters_in_filename += Learner.get_core_params_in_model_name(new_dictionary)
                parameters_in_filename += CMALearner.get_additional_params_in_model_name(new_dictionary)
                filename = "_".join([str(param) for param in parameters_in_filename]) + ".json"

                f = open("../" + filename, "w")
                dictionary_string = json.dumps(new_dictionary, indent=4)
                f.write(dictionary_string)
                f.close()

                # Calculate latest mean and save as genome file
                mean = es.x0
                fitness = -es.result.fbest
                intermediate_seed_name = CMALearner.get_model_name_from_dictionary(new_dictionary, fitness)
                self.save_genome(mean, intermediate_seed_name)

                sys.exit()

            #signal.signal(signal.SIGTERM, signal_handler)

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

            # Get fitnesses of agents
            agent_fitness_lists, team_specialisations = self.fitness_calculator.calculate_fitness_of_agent_population(agent_population, self.calculate_specialisation)

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

    def get_seed_genome(self):
        """
        Get the seed genome to be used as cma's mean. If getting it from a file, the appropriate seed genome is the one
        that has the same parameter values as the current experiment

        @return: Genome and its fitness
        """

        if self.parameter_dictionary['algorithm']['cma']['seeding_required'] == "True" or \
                self.parameter_dictionary['algorithm'] == "cma_with_seeding" or \
                self.parameter_dictionary['algorithm'] == "partialcma":
            dictionary_copy = copy.deepcopy(self.parameter_dictionary)

            if dictionary_copy['general']['algorithm_selected'] != "partialcma":
                dictionary_copy['general']['algorithm_selected'] = "rwg"
            else:
                dictionary_copy['general']['algorithm_selected'] = "cma"

            # If individual reward, allow seeds that used any number of agents
            if dictionary_copy['general']['reward_level'] == "individual":
                environment_name = dictionary_copy['general']['environment']
                dictionary_copy['environment'][environment_name]['num_agents'] = '*'

            # Looks for seedfiles with the same parameters as the current experiment
            parameters_in_name = Learner.get_core_params_in_model_name(dictionary_copy)

            possible_seedfiles = None

            seedfile_prefix = "_".join([str(param) for param in parameters_in_name])
            seedfile_extension = self.Agent.get_model_file_extension()
            possible_seedfiles = glob(f'{seedfile_prefix}*{seedfile_extension}')

            # Makes sure there is only one unambiguous seedfile
            if len(possible_seedfiles) == 0:
                raise RuntimeError('No valid seed files')
            elif len(possible_seedfiles) > 1:
                raise RuntimeError('Too many valid seed files')
            else:
                model_file_extension = self.Agent.get_model_file_extension()
                seed_fitness = float(possible_seedfiles[0].split("_")[-1].strip(model_file_extension))
                return self.Agent.load_model_from_file(possible_seedfiles[0]), seed_fitness

        else:
            # Set mini-rwg parameters
            temporary_parameter_dictionary = copy.deepcopy(self.parameter_dictionary)
            temporary_parameter_dictionary['general']['algorithm_selected'] = "rwg"

            env_name = temporary_parameter_dictionary['general']['environment']
            num_agents = temporary_parameter_dictionary['environment'][env_name]['num_agents']
            temporary_parameter_dictionary['algorithm']['agent_population_size'] = num_agents

            temporary_parameter_dictionary['algorithm']['rwg']['sampling_distribution'] = "normal"
            temporary_parameter_dictionary['algorithm']['rwg']['normal']['mean'] = 0
            temporary_parameter_dictionary['algorithm']['rwg']['normal']['std'] = 1

            # Write mini-rwg parameters to file
            temporary_parameter_file = f"temporary_parameter_file_{temporary_parameter_dictionary['general']['seed']}.json"
            f = open(temporary_parameter_file, "w")
            dictionary_string = json.dumps(temporary_parameter_dictionary, indent=4)
            f.write(dictionary_string)
            f.close()

            # Generate seed genome
            temporary_calculator = FitnessCalculator(temporary_parameter_file)
            learner = RWGLearner(temporary_calculator)
            seed_genome, seed_fitness = learner.learn(logging=False)

            # Delete temporary paramter file
            #os.remove(temporary_parameter_file)

            return seed_genome, seed_fitness

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
        model_name = self.generate_model_name(genome_fitness,)
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

        result_data = Learner.get_core_params_in_model_name(self.parameter_dictionary) + \
                      CMALearner.get_additional_params_in_model_name(self.parameter_dictionary) + \
                      [seed_fitness, genome_fitness, model_name]
        results = ",".join([str(element) for element in result_data])

        results_file.write(f"{results}\n")
        results_file.close()

    def generate_model_name(self, fitness):
        return CMALearner.get_model_name_from_dictionary(self.parameter_dictionary, fitness)

    @staticmethod
    def get_model_name_from_dictionary(parameter_dictionary, fitness):
        """
        Create a name string for a model generated using the given parameter file, its rank and fitness value

        @param fitness: Fitness of the model to be saved
        @return: The model name as a string
        """

        parameters_in_name = Learner.get_core_params_in_model_name(parameter_dictionary)
        parameters_in_name += CMALearner.get_additional_params_in_model_name(parameter_dictionary)

        # Get fitness
        parameters_in_name += [fitness]

        # Put all in a string and return
        return "_".join([str(param) for param in parameters_in_name])

    @staticmethod
    def get_additional_params_in_model_name(parameter_dictionary):
        """
        Return the parameters of the model that are specific to CMA-ES

        @param parameter_dictionary: Dictionary containing the desired parameters
        @return: List of parameter values
        """
        parameters_in_name = []

        # Get algorithm params for relevant algorithm
        parameters_in_name += [parameter_dictionary['algorithm']['agent_population_size']]
        parameters_in_name += [parameter_dictionary['algorithm']['cma']['sigma']]
        parameters_in_name += [parameter_dictionary['algorithm']['cma']['generations']]
        parameters_in_name += [parameter_dictionary['algorithm']['cma']['tolx']]
        parameters_in_name += [parameter_dictionary['algorithm']['cma']['tolfunhist']]

        return parameters_in_name

    @staticmethod
    def get_results_headings(parameter_dictionary):
        """
        Get a list containing (most) of the columns that will be printed to the results file

        @param parameter_dictionary: Dictionary containing parameter values
        @return: List of column names
        """
        headings = Learner.get_results_headings(parameter_dictionary)
        headings += ["agent_population_size",
                     "sigma",
                     "generations",
                     "tolx",
                     "tolfunhist"]

        return headings
