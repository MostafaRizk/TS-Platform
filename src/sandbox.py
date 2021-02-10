import argparse

from scripts.benchmarking import plot_envs_vs_NN_arch


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
                                 mean_lim=(-20000, 250000), var_lim=(0, 75000), dist_lim=(10 ** -1, 10000), spec_metric_key=m, num_samples=10000, num_episodes=20)


parser = argparse.ArgumentParser(description='Generate RWG Analysis Plots')
parser.add_argument('--genome_directory', action="store", dest="genome_directory")
parser.add_argument('--metric', action="store", dest="metric")
genome_directory = parser.parse_args().genome_directory
metric = parser.parse_args().metric
generate_plots(genome_directory, metric)