'''
If you ran cma, then ran partialcma to pick up where cma cut off, this script adjusts the model names and combines them into a results file
'''
import os

from glob import glob
from shutil import copyfile, move

def combine_results(directory):
    experiment_directory = os.path.join(directory, "experiments")
    results_directory = os.path.join(directory, "results")

    # For each experiment
    cma_json_files = glob(f'{experiment_directory}/cma*json')

    for cma_json in cma_json_files:
        # Find all npy files
        model_prefix = "_".join(str(param) for param in cma_json.split("/")[-1].strip(".json").strip("cma_with_seeding").split("_")[:-3])
        original_npy_files = glob(f"{results_directory}/cma_with_seeding*{model_prefix}*npy")
        partial_npy_files = glob(f"{results_directory}/partialcma*{model_prefix}*npy")

        # If the prefix is cma_with_seeding, change to cma and keep track of the biggest generation
        biggest_generation_for_seed = {}

        for file in original_npy_files:
            newfilename = "cma_" + "_".join(str(param) for param in file.split("/")[-1].split("_")[3:])
            seed = file.split("/")[-1].strip(".npy").split("_")[7]
            generation = file.split("/")[-1].strip(".npy").split("_")[-1]

            if generation != "final":
                generation = int(generation)

            if seed in biggest_generation_for_seed:
                if biggest_generation_for_seed[seed] != "final":
                    if generation == "final" or generation > biggest_generation_for_seed[seed]:
                        biggest_generation_for_seed[seed] = generation
            else:
                biggest_generation_for_seed[seed] = generation

            newfile = os.path.join(results_directory, newfilename)
            move(file, newfile)

        # If the prefix is partialcma, change to cma and add generation to biggest generation (unless it's final)
        for file in partial_npy_files:
            seed = file.split("/")[-1].strip(".npy").split("_")[5]
            generation = file.split("/")[-1].strip(".npy").split("_")[-1]

            if generation != "final":
                generation = int(generation)
                new_generation = biggest_generation_for_seed[seed] + generation
            else:
                new_generation = generation

            newfilename = "cma_" + "_".join(str(param) for param in file.split("/")[-1].strip(".npy").split("_")[1:-1]) + f"_{new_generation}.npy"
            newfile = os.path.join(results_directory, newfilename)
            move(file, newfile)

def change_generations(directory):
    '''
    Change the number of generations for each npy file to 1000
    @param directory:
    @return:
    '''

    npy_files = glob(f'{directory}/cma*npy')
    for file in npy_files:
        total_generations = file.split("/")[-1].strip(".npy").split("_")[-5]
        if total_generations != "1000":
            prefix = "_".join(str(param) for param in file.split("/")[-1].strip(".npy").split("_")[:-5])
            suffix = "_".join(str(param) for param in file.split("/")[-1].strip(".npy").split("_")[-4:])
            newfile = prefix + "_1000_" + suffix
            move(file, newfile)


directory = "/Users/mostafa/Documents/Code/PhD/TS-Platform/results/2021_02_17_cma_for_diff_slopes_combined"
#combine_results(directory)

results_directory = os.path.join(directory, "results")
change_generations(results_directory)