import argparse

from scripts.benchmarking import plot_envs_vs_NN_arch


def generate_plots(genome_directory, env, metric=None):
    #biases = [False, True]
    biases = [True]
    if metric is None:
        if env == "slope":
            metrics = ["R_coop", "R_coop_eff", "R_spec"]
        elif env == "tmaze":
            metrics = ["num_pairs"]
    else:
        metrics = [metric]

    for bias in biases:
        for m in metrics:
            if env == "slope":
                N_bins = [-20000, -6500, 7000, 20500, 34000, 47500, 61000, 74500, 88000, 101500, 115000,
                                         128500, 142000, 155500, 169000, 182500, 196000, 209500, 223000, 236500,
                                         250000]
                mean_lim = (-20000, 250000)
                var_lim = (0, 75000)
                dist_lim = (10 ** -1, 10000)
                num_samples = 10000
                num_episodes = 20

            elif env == "tmaze":
                N_bins = [n for n in range(0, 15, 5)]
                mean_lim = (0, 15)
                var_lim = (0, 15)
                dist_lim = (10 ** -1, 10000)
                num_samples = 10000
                num_episodes = 1

            plot_envs_vs_NN_arch(genome_directory, bias, env=env,
                                 N_bins=N_bins,
                                 mean_lim=mean_lim, var_lim=var_lim, dist_lim=dist_lim, spec_metric_key=m, num_samples=num_samples, num_episodes=num_episodes)


parser = argparse.ArgumentParser(description='Generate RWG Analysis Plots')
parser.add_argument('--genome_directory', action="store", dest="genome_directory")
parser.add_argument('--env', action="store", dest="env")
parser.add_argument('--metric', action="store", dest="metric")
genome_directory = parser.parse_args().genome_directory
env = parser.parse_args().env
metric = parser.parse_args().metric
generate_plots(genome_directory, env, metric)