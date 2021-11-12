import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
import argparse

from glob import glob

# In the model name for a decentralised team separated by '_', what index contains the id of the agent
id_index = -3


def plot_weight_distribution(evolved_genomes_directory, results_file, graph_file, num_agents=2):
    """
    Create boxplot showing the distribution of weight values for each weight in a genome, over a collection of genomes.

    @param results_file: CSV file containing the results of several successful evolutionary runs.
    @param graph_file: The name to save the plot as
    @return:
    """
    # Load all genomes into pandas dataframe
    evolved_data = pd.read_csv(os.path.join(evolved_genomes_directory, f"data/{results_file}"))

    matrix = []

    # For each genome
    for index, row in evolved_data.iterrows():
        # Check if it has team reward and 2 agents
        if row["learning_type"] == "centralised" and row["reward_level"] == "team" and row["num_agents"] == num_agents:
            # Get genome from corresponding model file
            model_file = os.path.join(evolved_genomes_directory, row["model_name"])
            genome = np.load(model_file)
            # Add genome to matrix
            matrix += [genome]

        elif row["learning_type"] == "decentralised" and row["reward_level"] == "individual" and row["num_agents"] == num_agents:
            genome = np.array([])
            model_list = row["model_name"].split("_")
            model_prefix = model_list[:-3]
            model_suffix = model_list[-2:]
            model_name = "_".join(model_prefix + ['*'] + model_suffix)
            all_genome_files = glob(os.path.join(evolved_genomes_directory, model_name))

            for model_file in all_genome_files:
                genome_part = np.load(model_file)
                genome = np.concatenate((genome, genome_part))

            matrix += [genome]

    num_dimensions = len(matrix[0])

    # Plot data
    fig1, ax1 = plt.subplots(figsize=(12, 4))
    ax1.set_title('Distribution of Neural Network Weights')
    #ax1.set_ylim(-1000, 1000)
    ax1.set_ylabel('Weight Value')
    ax1.set_xlabel('Weight ID')

    x_range = np.arange(num_dimensions)

    #y = np.transpose(np.array(matrix))
    y=np.array(matrix)

    ax1.boxplot(y)

    #plt.show()
    plt.savefig(graph_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot weight distribution of a bunch of models')
    parser.add_argument('--evolved_genomes_directory', action="store")
    parser.add_argument('--results_file', action="store")
    parser.add_argument('--plot_file', action="store")

    evolved_genomes_directory = parser.parse_args().evolved_genomes_directory
    results_file = parser.parse_args().results_file
    plot_file = parser.parse_args().plot_file

    plot_weight_distribution(evolved_genomes_directory, results_file, plot_file)