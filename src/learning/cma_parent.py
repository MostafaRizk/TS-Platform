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

from learning.rwg_centralised import CentralisedRWGLearner
from learning.rwg_fully_centralised import FullyCentralisedRWGLearner


class CMALearner(Learner):
    def __init__(self, calculator):
        super().__init__(calculator)

        if self.parameter_dictionary['general']['algorithm_selected'] != "cma":
            raise RuntimeError(f"Cannot run cma. Parameters request "
                               f"{self.parameter_dictionary['general']['algorithm_selected']}")

        if self.parameter_dictionary['algorithm']['cma']['multithreading'] == "True":
            self.multithreading = True
        elif self.parameter_dictionary['algorithm']['cma']['multithreading'] == "False":
            self.multithreading = False
        else:
            self.multithreading = False
            raise RuntimeWarning("Multithreading setting not specified in parameters, defaulting to False (i.e. sequential execution)")

        # Log every x many generations
        self.logging_rate = self.parameter_dictionary['algorithm']['cma']['logging_rate']

    def get_seed_genome(self):
        """
        Get the seed genome to be used as cma's mean. If getting it from a file, the appropriate seed genome is the one
        that has the same parameter values as the current experiment

        @return: Genome and its fitness
        """

        if self.parameter_dictionary['algorithm']['cma']['seeding_required'] == "True":
            dictionary_copy = copy.deepcopy(self.parameter_dictionary)

            if dictionary_copy['algorithm']['cma']['partial'] == "False":
                dictionary_copy['general']['algorithm_selected'] = "rwg"
            elif dictionary_copy['algorithm']['cma']['partial'] == "True":
                dictionary_copy['general']['algorithm_selected'] = "cma"
            else:
                raise RuntimeError("The value for partial cma is neither True nor False")

            # If individual reward, look for seeds that use just one agent
            # TODO: Should the agent learn in an environment with multiple other agents?
            if dictionary_copy['general']['reward_level'] == "individual":
                environment_name = dictionary_copy['general']['environment']
                dictionary_copy['environment'][environment_name]['num_agents'] = '1'

            # If this is a decentralised cma learner, use a centralised rwg seed
            if dictionary_copy['general']['learning_type'] == "decentralised":
                dictionary_copy['general']['learning_type'] = "centralised"

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

            if self.learning_type == "fully-centralised":
                temporary_parameter_dictionary['general']['learning_type'] = "fully-centralised"
            else:
                temporary_parameter_dictionary['general']['learning_type'] = "centralised"

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

            if self.learning_type != "fully-centralised":
                learner = CentralisedRWGLearner(temporary_calculator)

            else:
                learner = FullyCentralisedRWGLearner(temporary_calculator)

            seed_genome, seed_fitness = learner.learn(logging=False)

            # Delete temporary paramter file
            #os.remove(temporary_parameter_file)

            return seed_genome, seed_fitness

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
        parameters_in_name += [parameter_dictionary['algorithm']['cma']['tolflatfitness']]
        parameters_in_name += [parameter_dictionary['algorithm']['cma']['tolfun']]

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
                     "tolfunhist",
                     "tolflatfitness"
                     "tolfun"]

        return headings
