import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np

def plot_weight_distribution(results_file, graph_file):
    """
    Create boxplot showing the distribution of weight values for each weight in a genome, over a collection of genomes.

    @param results_file: CSV file containing the results of several successful evolutionary runs.
    @param graph_file: The name to save the plot as
    @return:
    """
    # Load all genomes into pandas dataframe
    evolved_data = pd.read_csv(results_file)

    matrix = []

    # For each genome
    for index, row in evolved_data.iterrows():
        # Check if it has team reward and 2 agents
        if row["reward_level"] == "team" and row["num_agents"] == 2:
            # Get genome from corresponding model file
            model_file = os.path.join(evolved_genomes_directory, row["model_name"])
            genome = np.load(model_file)
            # Add genome to matrix
            matrix += [genome]

        ''''''
        # Duplicate genome if it has an individual reward
        if row["reward_level"] == "individual" and row["num_agents"] == 2:
            # Get genome from corresponding model file
            model_file = os.path.join(evolved_genomes_directory, row["model_name"])
            genome = np.load(model_file)
            # Add genome to matrix
            matrix += [np.append(genome, genome)]

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

evolved_genomes_directory = "/Users/mostafa/Documents/Code/PhD/TS-Platform/results/2020_11_24_magic_plot_combined_new_seed/results"
generation = "final"
results_file = os.path.join(evolved_genomes_directory, f"results_{generation}.csv")
graph_file = "weight_distribution.png"
plot_weight_distribution(results_file, graph_file)