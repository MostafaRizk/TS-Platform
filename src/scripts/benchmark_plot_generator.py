import os
import argparse

from scripts.benchmarking import BenchmarkPlotter
from scripts.benchmarking import get_experiment_name_from_filename
from scripts.benchmarking import plot_envs_vs_NN_arch
from glob import glob

parser = argparse.ArgumentParser(description='Generate RWG Analysis Plots')
parser.add_argument('--genome_directory', action="store", dest="genome_directory")
genome_directory = parser.parse_args().genome_directory

os.chdir(genome_directory)
csv_files = glob(f'*.csv')
experiments = []

for filename in csv_files:
    shortened_name = get_experiment_name_from_filename(filename)
    experiments += [(shortened_name, filename)]

original_dir = os.getcwd()

for experiment in experiments:
    experiment_name = experiment[0]
    experiment_filename = experiment[1]

    plotter = BenchmarkPlotter(experiment_name, experiment_filename)
    plotter.save_all_sample_stats(
        N_bins=[-20000, -10000, 0, 10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000, 110000, 120000, 130000, 140000, 150000, 160000, 170000, 180000, 190000, 200000],
                                  mean_lim=(-20000, 200000), var_lim=(0, 60000), dist_lim=(10**-1, 10000), num_samples=10000, num_episodes=5)


def generate_plots(genome_directory, metric=None):
    biases = [False, True]
    if metric is None:
        metrics = ["R_coop", "R_coop_eff", "R_spec"]
    else:
        metrics = [metric]

    for bias in biases:
        for m in metrics:
            plot_envs_vs_NN_arch(genome_directory, bias,
                                 N_bins=[-20000, -6500, 7000, 20500, 34000, 47500, 61000, 74500, 88000, 101500, 115000,
                                         128500, 142000, 155500, 169000, 182500, 196000, 209500, 223000, 236500,
                                         250000],
                                 mean_lim=(-20000, 250000), var_lim=(0, 75000), dist_lim=(10 ** -1, 10000), spec_metric_key=m, num_samples=10000, num_episodes=5)



