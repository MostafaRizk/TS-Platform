from learning.learner_parent import Learner
from helpers import novelty_helpers
from functools import partial


class GALearner(Learner):
    def __init__(self, calculator):
        super().__init__(calculator)

        if self.parameter_dictionary['general']['algorithm_selected'] != "ga":
            raise RuntimeError(f"Cannot run ga. Parameters request "
                               f"{self.parameter_dictionary['general']['algorithm_selected']}")

        if self.parameter_dictionary['algorithm']['cma']['multithreading'] == "True":
            self.multithreading = True
        elif self.parameter_dictionary['algorithm']['cma']['multithreading'] == "False":
            self.multithreading = False
        else:
            self.multithreading = False
            raise RuntimeWarning(
                "Multithreading setting not specified in parameters, defaulting to False (i.e. sequential execution)")

        # Log every x many generations
        self.logging_rate = self.parameter_dictionary['algorithm']['ga']['logging_rate']
        self.calculate_behaviour_distance = partial(novelty_helpers.calculate_distance, metric=self.novelty_params['distance_metric'])

    @staticmethod
    def get_additional_params_in_model_name(parameter_dictionary):
        """
        Return the parameters of the model that are specific to GA

        @param parameter_dictionary: Dictionary containing the desired parameters
        @return: List of parameter values
        """
        parameters_in_name = []

        # Get algorithm params for relevant algorithm
        parameters_in_name += [parameter_dictionary['algorithm']['agent_population_size']]
        parameters_in_name += [parameter_dictionary['algorithm']['ga']['generations']]
        parameters_in_name += [parameter_dictionary['algorithm']['ga']['mutation_probability']]
        parameters_in_name += [parameter_dictionary['algorithm']['ga']['tournament_size']]
        parameters_in_name += [parameter_dictionary['algorithm']['ga']['mu']]
        parameters_in_name += [parameter_dictionary['algorithm']['ga']['sigma']]
        parameters_in_name += [parameter_dictionary['algorithm']['ga']['num_parents']]
        parameters_in_name += [parameter_dictionary['algorithm']['ga']['num_children']]
        parameters_in_name += [parameter_dictionary['algorithm']['ga']['crowding_factor']]

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
                     "generations",
                     "mutation_probability",
                     "tournament_size",
                     "mu",
                     "sigma",
                     "num_parents",
                     "num_children",
                     "crowding_factor"]

        return headings