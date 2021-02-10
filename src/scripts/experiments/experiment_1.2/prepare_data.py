import argparse
import os
import shutil

from glob import glob

def create_folders(parent_directory, slopes):
    # Create slope folders with 'results' and 'analysis' sub-folders
    for slope in slopes:
        slope_dir = os.path.join(parent_directory, f'slope_{slope}')
        if not os.path.exists(slope_dir):
            os.mkdir(slope_dir)

        results_dir = os.path.join(slope_dir, 'results')
        analysis_dir = os.path.join(slope_dir, 'analysis')

        if not os.path.exists(results_dir):
            os.mkdir(results_dir)

        if not os.path.exists(analysis_dir):
            os.mkdir(analysis_dir)

def move_results(parent_directory):
    # Move csv files
    csv_files = glob(f'{parent_directory}/all_genomes*csv')
    for file in csv_files:
        slope = file.split("/")[-1].split("_")[11] # The slope (sliding speed) is the 11th item in the name
        original_path = os.path.join(parent_directory, file)
        new_path = os.path.join(parent_directory, f'slope_{slope}/results/{file.split("/")[-1]}')
        shutil.move(original_path, new_path)

    # Move npy files
    npy_files = glob(f'{parent_directory}/rwg*npy')
    for file in npy_files:
        slope = file.split("/")[-1].split("_")[9]  # The slope (sliding speed) is the 9th item in the name
        original_path = os.path.join(parent_directory, file)
        new_path = os.path.join(parent_directory, f'slope_{slope}/results/{file.split("/")[-1]}')
        shutil.move(original_path, new_path)

parser = argparse.ArgumentParser(description='Move results files into relevant folders')
parser.add_argument('--results_directory', action="store", dest="results_directory")
parent_directory = parser.parse_args().results_directory

slopes = [0, 2, 4]
create_folders(parent_directory, slopes)
move_results(parent_directory)

