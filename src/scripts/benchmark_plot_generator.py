import os

from scripts.benchmarking import BenchmarkPlotter
from glob import glob

csv_files = glob(f'result_files/*.csv')
experiments = []

for filename in csv_files:
    shortened_name = filename.split("/")[1].strip(".csv")
    experiments += [(shortened_name, filename)]

original_dir = os.getcwd()

for experiment in experiments:
    experiment_name = experiment[0]
    experiment_filename = experiment[1]

    plotter = BenchmarkPlotter(experiment_name, experiment_filename)
    #plotter.save_all_sample_stats(N_bins=[-5000, 0, 5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000, 45000, 50000],
    #                              mean_lim=(-3000, 50000), var_lim=(0, 15000), dist_lim=(1, 100000))
    #plotter.save_all_sample_stats(
    #    N_bins=[0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
    #                              mean_lim=(0, 50), var_lim=(0, 15), dist_lim=(1, 100))

    plotter.save_all_sample_stats(
        N_bins=[-10000, -5000, 0, 5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000],
                                  mean_lim=(-10000, 40000), var_lim=(0, 15000), dist_lim=(10**-1, 10000))
