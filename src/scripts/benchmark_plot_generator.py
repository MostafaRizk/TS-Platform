import os

from scripts.benchmarking import BenchmarkPlotter
from glob import glob

#csv_files = glob(f'../../results/2020_12_18_rwg_analysis_smooth_fitness/results/*.csv')
csv_files = glob(f'../../results/2020_09_04_Identical_oller_comparison/results/*.csv')
experiments = []

for filename in csv_files:
    shortened_name = filename.split("/")[5].strip(".csv")
    experiments += [(shortened_name, filename)]

original_dir = os.getcwd()

for experiment in experiments:
    experiment_name = experiment[0]
    experiment_filename = experiment[1]

    plotter = BenchmarkPlotter(experiment_name, experiment_filename)

    plotter.save_all_sample_stats(
        N_bins=[-20000, -10000, 0, 10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000, 110000, 120000, 130000, 140000, 150000, 160000, 170000, 180000, 190000, 200000],
                                  mean_lim=(-20000, 200000), var_lim=(0, 60000), dist_lim=(10**-1, 10000))
