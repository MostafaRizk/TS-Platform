import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

from helpers import sammon
from mpl_toolkits import mplot3d
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from evaluation.plot_scalability import from_string
from operator import add
from glob import glob
from matplotlib.colors import ListedColormap

seed = 1
sliding_speed = 8
num_agents = 2 # Only supports 2 agents
spec_score_keys = ["R_coop", "R_coop_eff", "R_spec", "R_coop x P", "R_coop_eff x P", "R_spec x P"]
chosen_key = 0 # Chosen spec metric
spec_scores = {}
for key in spec_score_keys:
    spec_scores[key] = []


def generate_matrix(rwg_genomes_file, evolved_genomes_directory, results_file, num_episodes):
    rwg_data = []
    f = open(rwg_genomes_file, "r")
    rwg_data += f.read().strip().split("\n")

    N_episodes = num_episodes
    matrix = []
    scores = []
    rwg_indices = [i for i in range(120)] + [i for i in range(-120,0)]

    num_rwg = 0
    num_ctde = 0
    num_fd = 0

    # Number of scores (fitness and specialisation)
    num_spec_scores = len(spec_score_keys) * N_episodes
    num_scores = N_episodes + num_spec_scores

    # Get rwg data
    for index in rwg_indices:
        row = rwg_data[index]
        genome = [float(element) for element in row.split(",")[0:-num_scores]]
        episode_scores = [float(score) for score in row.split(",")[-num_scores:-num_spec_scores]]

        # Make a list of lists. For each specialisation metric, there is a list of scores for all episodes
        raw_spec_scores = row.split(",")[-num_spec_scores:]
        raw_spec_scores = [[float(raw_spec_scores[i + j]) for i in range(0, len(raw_spec_scores), len(spec_score_keys))] for j in range(len(spec_score_keys))]

        # Take the average of all the episodes for each metric and store them in the dictionary
        ''''''
        for i in range(len(spec_score_keys)):
            spec_scores[spec_score_keys[i]] += [np.mean(raw_spec_scores[i])]

        mean_score = np.mean(episode_scores)
        scores += [mean_score]

        matrix += [genome]
        num_rwg += 1

    # Get evolved data
    evolved_genomes_directory = os.path.join(evolved_genomes_directory, "data")
    ''''''
    for generation in ["final"]:
        results_file = os.path.join(evolved_genomes_directory, results_file)
        # Load all genomes into pandas dataframe
        evolved_data = pd.read_csv(results_file)
        # For each genome
        for index, row in evolved_data.iterrows():
            # Check if it has team reward and 2 agents
            if row["num_agents"] == num_agents and row["sliding_speed"] == sliding_speed:
                ''''''
                if row["reward_level"] == "team" and row["learning_type"] == "centralised":
                    # Get genome from corresponding model file
                    model_file = os.path.join(evolved_genomes_directory, row["model_name"])
                    genome = np.load(model_file)
                    # Add genome to matrix
                    matrix += [genome]
                    fitness = from_string(row["fitness"])[0:N_episodes]
                    specialisation = from_string(row["specialisation"])[0:N_episodes]

                    team_fitness_list = [0] * len(fitness[0])

                    for j in range(num_agents):
                        team_fitness_list = list(map(add, team_fitness_list, fitness[j]))

                    scores += [np.mean(team_fitness_list)]

                    for key_index in range(len(spec_score_keys)):
                        specialisation_for_key = [episode[key_index] for episode in specialisation]
                        spec_scores[spec_score_keys[key_index]] += [np.mean(specialisation_for_key)]

                    num_ctde += 1


                ''''''
                if row["reward_level"] == "individual" and row["learning_type"] == "decentralised":
                    # Get genome from corresponding model file
                    agent_1_model = row["model_name"].split("_")
                    agent_1_model[-3] = "0"
                    agent_1_model[-2] = "*"
                    agent_1_model = "_".join(agent_1_model)

                    agent_2_model = row["model_name"].split("_")
                    agent_2_model[-2] = "*"
                    agent_2_model = "_".join(agent_2_model)

                    agent_1_models = glob(agent_1_model)
                    agent_2_models = glob(agent_2_model)

                    assert len(agent_1_models) == len(agent_2_models) == 1, "There must be 1 model for each agent"

                    model_file_1 = os.path.join(evolved_genomes_directory, agent_1_models[0])
                    model_file_2 = os.path.join(evolved_genomes_directory, agent_2_models[0])
                    genome = np.concatenate((np.load(model_file_1), np.load(model_file_2)))

                    # Add genome to matrix
                    matrix += [genome]
                    fitness = from_string(row["fitness"])[0:N_episodes]
                    specialisation = from_string(row["specialisation"])[0:N_episodes]

                    team_fitness_list = [0] * len(fitness[0])

                    for j in range(num_agents):
                        team_fitness_list = list(map(add, team_fitness_list, fitness[j]))

                    scores += [np.mean(team_fitness_list)]

                    for key_index in range(len(spec_score_keys)):
                        specialisation_for_key = [episode[key_index] for episode in specialisation]
                        spec_scores[spec_score_keys[key_index]] += [np.mean(specialisation_for_key)]

                    num_fd += 1


    scores = np.array(scores)

    return matrix, scores


def plot_dimensionality_reduction(matrix, scores, plot_destination):
    # Do dimensionality reduction using PCA
    '''
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(np.array(matrix))
    print(np.sum(pca.explained_variance_ratio_))
    x = pca_result[:,0]
    y = pca_result[:,1]
    '''
    # Do dimensionality reduction using t-SNE
    ''''''
    random_state = np.random.RandomState(seed)
    tsne = TSNE(n_components=2, verbose=1, perplexity=40.0, random_state=random_state)
    tsne_results = tsne.fit_transform(np.array(matrix))
    # tsne_results = tsne.fit_transform(pca_result)
    x = tsne_results[:, 0]
    y = tsne_results[:, 1]
    z = scores

    fig = plt.figure()
    #cm = plt.cm.get_cmap('RdYlGn')

    vermillion = (213, 94, 0)
    bluish_green = (0, 158, 115)
    generalist_colour = vermillion
    specialist_colour = bluish_green
    # Divide RGB value by 256 to map it to 0-1 range
    N = 256
    vals = np.ones((N, 4))
    vals[:, 0] = np.linspace(generalist_colour[0] / N, specialist_colour[0] / N, N)
    vals[:, 1] = np.linspace(generalist_colour[1] / N, specialist_colour[1] / N, N)
    vals[:, 2] = np.linspace(generalist_colour[2] / N, specialist_colour[2] / N, N)
    # Create custom colour map
    cm = ListedColormap(vals)

    key = spec_score_keys[chosen_key]
    ax = fig.add_subplot(projection='3d')
    ax.scatter3D(x, y, z, c=spec_scores[key], vmin=0, vmax=1, cmap=cm)
    #ax.colorbar()
    plt.title("3D Mapping of Fitness Landscape")
    # fig.legend(loc="lower left")
    # plt.savefig(f"tsne_new_spec_lhs_5_60_samples.png")
    plt.savefig(plot_destination)
    #plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Dimensionality Reduction')
    parser.add_argument('--rwg_genomes_file', action="store")
    parser.add_argument('--evolved_genomes_directory', action="store")
    parser.add_argument('--results_file', action="store")
    parser.add_argument('--num_episodes', action="store")
    parser.add_argument('--plot_destination', action="store")

    rwg_genomes_file = parser.parse_args().rwg_genomes_file
    evolved_genomes_directory = parser.parse_args().evolved_genomes_directory
    results_file = parser.parse_args().results_file
    num_episodes = int(parser.parse_args().num_episodes)
    plot_destination = parser.parse_args().plot_destination

    matrix, scores = generate_matrix(rwg_genomes_file, evolved_genomes_directory, results_file, num_episodes)
    plot_dimensionality_reduction(matrix, scores, plot_destination)


