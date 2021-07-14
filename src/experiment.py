import argparse
import copy
import json

from fitness import FitnessCalculator
from learning.learner_parent import Learner
from learning.rwg_centralised import CentralisedRWGLearner
from learning.rwg_fully_centralised import FullyCentralisedRWGLearner
from learning.cma_centralised import CentralisedCMALearner
from learning.cma_decentralised import DecentralisedCMALearner
from learning.cma_fully_centralised import FullyCentralisedCMALearner
from learning.ga_centralised import CentralisedGALearner


def run_experiment(parameter_filename):
    parameter_dictionary = json.loads(open(parameter_filename).read())
    fitness_calculator = FitnessCalculator(parameter_filename)

    if parameter_dictionary["general"]["algorithm_selected"] == "rwg":
        if parameter_dictionary["general"]["learning_type"] == "centralised":
            learner = CentralisedRWGLearner(fitness_calculator)
        elif parameter_dictionary["general"]["learning_type"] == "fully-centralised":
            learner = FullyCentralisedRWGLearner(fitness_calculator)
        else:
            raise RuntimeError("Invalid learning type for rwg")

        genome, fitness = learner.learn()

    elif parameter_dictionary["general"]["algorithm_selected"] == "cma":

        if parameter_dictionary["algorithm"]["cma"]["seeding_included"] == "True":
            #if parameter_dictionary["general"]["learning_type"] == "fully-centralised":
            #    raise RuntimeError("Seeding included is not yet supported for fully centralised")

            # Load default rwg parameters
            if parameter_dictionary["general"]["environment"] == "slope":
                default_rwg_parameter_filename = 'default_slope_rwg_parameters_individual.json'

            rwg_parameter_dictionary = json.loads(open(default_rwg_parameter_filename).read())

            # Copy general parameters from cma to rwg
            rwg_parameter_dictionary["general"] = copy.deepcopy(parameter_dictionary["general"])
            rwg_parameter_dictionary["general"]["algorithm_selected"] = "rwg"

            if parameter_dictionary["general"]["learning_type"] == "decentralised":
                rwg_parameter_dictionary["general"]["learning_type"] = "centralised"

            if parameter_dictionary["general"]["reward_level"] == "team":
                rwg_parameter_dictionary["general"]["reward_level"] = "team"
                environment_name = parameter_dictionary["general"]["environment"]

                if parameter_dictionary["general"]["team_type"] == "heterogeneous":
                    rwg_parameter_dictionary["algorithm"]["agent_population_size"] *= parameter_dictionary["environment"][environment_name]["num_agents"]

            # Copy environment parameters from cma to rwg
            rwg_parameter_dictionary["environment"] = copy.deepcopy(parameter_dictionary["environment"])

            if rwg_parameter_dictionary["general"]["reward_level"] == "individual":
                environment_name = rwg_parameter_dictionary["general"]["environment"]
                rwg_parameter_dictionary["environment"][environment_name]["num_agents"] = 1

            # Copy agent parameters from cma to rwg
            rwg_parameter_dictionary["agent"] = copy.deepcopy(parameter_dictionary["agent"])

            # Create rwg json file and load to fitness calculator
            parameters_in_name = Learner.get_core_params_in_model_name(rwg_parameter_dictionary)
            parameters_in_name += CentralisedRWGLearner.get_additional_params_in_model_name(rwg_parameter_dictionary)
            new_rwg_parameter_filename = "_".join([str(param) for param in parameters_in_name]) + ".json"
            f = open(new_rwg_parameter_filename, "w")
            rwg_dictionary_string = json.dumps(rwg_parameter_dictionary, indent=4)
            f.write(rwg_dictionary_string)
            f.close()
            rwg_fitness_calculator = FitnessCalculator(new_rwg_parameter_filename)

            # Seeding
            learner1 = CentralisedRWGLearner(rwg_fitness_calculator)
            genome1, fitness1 = learner1.learn()

            # Learning
            if parameter_dictionary["general"]["learning_type"] == "centralised":
                learner2 = CentralisedCMALearner(fitness_calculator)
                genome2, fitness2 = learner2.learn()

            elif parameter_dictionary["general"]["learning_type"] == "decentralised":
                learner2 = DecentralisedCMALearner(fitness_calculator)
                genomes, fitnesses = learner2.learn()
        else:
            if parameter_dictionary["general"]["learning_type"] == "centralised":
                learner = CentralisedCMALearner(fitness_calculator)
                genome, fitness = learner.learn()

            elif parameter_dictionary["general"]["learning_type"] == "decentralised":
                learner = DecentralisedCMALearner(fitness_calculator)
                genomes, fitnesses = learner.learn()

            elif parameter_dictionary["general"]["learning_type"] == "fully-centralised":
                learner = FullyCentralisedCMALearner(fitness_calculator)
                genomes, fitnesses = learner.learn()

    elif parameter_dictionary["general"]["algorithm_selected"] == "ga":
        if parameter_dictionary["general"]["learning_type"] == "centralised":
            learner = CentralisedGALearner(fitness_calculator)
            genome, fitness = learner.learn()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run an experiment')
    parser.add_argument('--parameters', action="store", dest="parameter_filename")
    parameter_filename = parser.parse_args().parameter_filename

    run_experiment(parameter_filename)
