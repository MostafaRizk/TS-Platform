import os

from scripts.benchmarking import BenchmarkPlotter, plot_envs_vs_NN_arch
from glob import glob

og_dir = f'/Users/mostafa/Documents/Code/PhD/TS-Platform/results/2020_09_04_Identical_oller_comparison/analysis/'
new_dir = f'/Users/mostafa/Documents/Code/PhD/TS-Platform/results/2020_12_18_rwg_analysis_smooth_fitness/analysis/'

dirs = [og_dir, new_dir]
biases = [False, True]

for dir in dirs:
    for bias in biases:
        plot_envs_vs_NN_arch(dir, bias,
                     N_bins=[-20000, -6500, 7000, 20500, 34000, 47500, 61000, 74500, 88000, 101500, 115000, 128500, 142000, 155500, 169000, 182500, 196000, 209500, 223000, 236500, 250000],
                     mean_lim=(-20000, 250000), var_lim=(0, 75000), dist_lim=(10**-1, 10000))
