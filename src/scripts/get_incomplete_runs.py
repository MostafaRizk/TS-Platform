import os
import pandas as pd
import json

from glob import glob
from learning.cma import CMALearner
from learning.learner_parent import Learner
from shutil import copyfile, move


def get_incomplete_runs(old_directory, new_directory):
    # For each cma json file
    old_experiments_directory = os.path.join(old_directory, 'experiments')
    old_results_directory = os.path.join(old_directory, 'results')

    new_experiments_directory = os.path.join(new_directory, 'experiments')

    cma_json_files = glob(f'{old_experiments_directory}/cma*json')

    new_experiments_list = os.path.join(new_experiments_directory, "LIST_cma")
    g = open(new_experiments_list, "a")

    for cma_json in cma_json_files:
        # If it doesn't have a final model
        model_prefix = cma_json.split("/")[-1].strip(".json")
        final_models = glob(f"{old_results_directory}/{model_prefix}*_final.npy")

        if len(final_models) > 1:
            raise RuntimeError("Too many matching final models")

        elif len(final_models) < 1:
            # Find model with highest generation
            other_generations = glob(f"{old_results_directory}/{model_prefix}*.npy")
            latest_model = None
            latest_generation = 0

            for model_name in other_generations:
                generation = int(model_name.split("/")[-1].strip(".npy").split("_")[-1])
                if generation > latest_generation:
                    latest_generation = generation
                    latest_model = model_name

            # Create new version of json with 'partialcma' in filename and dictionary
            if latest_model is None:
                raise RuntimeError("No saved models. Evolution didn't make it to 20 generations")
            else:
                # Create modified parameter dictionary for new run
                parameter_dictionary = json.loads(open(cma_json).read())
                parameter_dictionary["general"]["algorithm_selected"] = "partialcma"
                parameter_dictionary["algorithm"]["cma"]["generations"] -= latest_generation
                parameters_in_filename = []
                parameters_in_filename += Learner.get_core_params_in_model_name(parameter_dictionary)
                parameters_in_filename += CMALearner.get_additional_params_in_model_name(parameter_dictionary)

                # Turn parameter dictionary into json file in new directory
                filename = "_".join([str(param) for param in parameters_in_filename]) + ".json"
                new_parameter_file = os.path.join(new_experiments_directory, filename)
                f = open(new_parameter_file, "w")
                dictionary_string = json.dumps(parameter_dictionary, indent=4)
                f.write(dictionary_string)
                f.close()

                # Add experiment to new LIST file
                g.write(f"python3 experiment.py --parameters {filename}\n")

                # Copy model to new directory
                filename = os.path.join(new_experiments_directory, latest_model.split("/")[-1])
                copyfile(latest_model, filename)

    g.close()

def rename_as_cma(directory):
    npy_files = glob(f'{directory}/*npy')

    for file in npy_files:
        newfilename = "cma_heterogeneous_" + "_".join(str(param) for param in file.split("/")[-1].split("_")[1:])
        newfile = os.path.join(directory, newfilename)
        move(file, newfile)

results_path = "../../results/"
old_results_folder = "2021_02_12_evo_for_diff_slopes"
new_experiments_folder = "2021_02_15_partial_cma_for_incomplete_diff_slope_runs"

old_directory = os.path.join(results_path, old_results_folder)
new_directory = os.path.join(results_path, new_experiments_folder)

#get_incomplete_runs(old_directory, new_directory)
#rename_as_cma(os.path.join(new_directory, "experiments"))