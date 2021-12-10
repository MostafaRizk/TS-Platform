import argparse

from scripts.benchmarking import plot_envs_vs_NN_arch


def generate_plots(genome_directory, metric=None):
    biases = [False, True]
    if metric is None:
        # metrics = ["R_coop", "R_coop_eff", "R_spec"]
        metrics = ["R_coop"]
    else:
        metrics = [metric]

    for learning_type in ["centralised"]:
        for bias in biases:
            for m in metrics:
                for arch in ["FFNN", "RNN"]:
                    # Proper code
                    plot_envs_vs_NN_arch(genome_directory, bias, N_bins=list(range(-2000, 45000, 10000)), mean_lim=(-2000, 45000), var_lim=(0, 15000), dist_lim=(10 ** -1, 20000), spec_metric_key=m, num_samples=10000, num_episodes=20, env="slope", learning_type=learning_type, arch=arch)

                    # Dummy code for testing
                    #plot_envs_vs_NN_arch(genome_directory, bias, N_bins=list(range(-2000, 45000, 10000)), mean_lim=(-2000, 45000), var_lim=(0, 15000), dist_lim=(10 ** -1, 20000), spec_metric_key=m, num_samples=50, num_episodes=5, env="slope", learning_type=learning_type, arch=arch)

parser = argparse.ArgumentParser(description='Generate RWG Analysis Plots')
parser.add_argument('--genome_directory', action="store", dest="genome_directory")
parser.add_argument('--metric', action="store", dest="metric")
genome_directory = parser.parse_args().genome_directory
metric = parser.parse_args().metric
generate_plots(genome_directory, metric)