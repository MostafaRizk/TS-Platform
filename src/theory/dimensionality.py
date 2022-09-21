import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt

from fitness import FitnessCalculator
from agents.nn_agent_lean import NNAgent
from sklearn.decomposition import PCA
from matplotlib.colors import ListedColormap
from sklearn.cluster import KMeans
from tslearn.clustering import TimeSeriesKMeans

seed = 1
sliding_speed = 8
num_agents = 2  # Only supports 2 agents
num_episodes = 20
spec_score_keys = ["R_coop", "R_coop_eff", "R_spec", "R_coop x P", "R_coop_eff x P", "R_spec x P"]
chosen_key = 0  # Chosen spec metric


def generate_trajectory_data(rwg_genomes_file, parameter_path, save_file):
    rwg_data = []
    f = open(rwg_genomes_file, "r")
    rwg_data += f.read().strip().split("\n")
    f.close()

    N_episodes = num_episodes
    matrix = []
    scores = []
    spec_scores = []
    rwg_indices = [i for i in range(len(rwg_data))]
    #rwg_indices = [i for i in range(10)]

    # Number of scores (fitness and specialisation)
    num_spec_scores = len(spec_score_keys) * N_episodes
    num_scores = N_episodes + num_spec_scores

    fitness_calculator = FitnessCalculator(parameter_path)

    for index in rwg_indices:
        # Get genomes from RWG
        row = rwg_data[index]
        team_genome = [float(element) for element in row.split(",")[:-num_scores]]
        agent_list = []

        for i in range(num_agents):
           start = i * int(len(team_genome) / num_agents)
           end = (i + 1) * int(len(team_genome) / num_agents)
           sub_genome = team_genome[start:end]
           agent = NNAgent(fitness_calculator.get_observation_size(), fitness_calculator.get_action_size(),
                            parameter_path, sub_genome)
           agent_list.append(agent)

        # Re-evaluate and log trajectory, individual fitness and team spec for each agent
        results = fitness_calculator.calculate_fitness(controller_list=agent_list, render=False,
                                                       measure_specialisation=True, logging=False, logfilename=None,
                                                       render_mode="human")

        for i in range(num_agents):
            for ep in range(N_episodes):
                matrix.append(results["trajectory_matrix"][i][ep])
                scores.append(results["fitness_matrix"][i][ep])
                spec_scores.append(results["specialisation_list"][ep][chosen_key])

    f = open(save_file, "w")
    f.write("Trajectory, Fitness, Specialisation\n")

    for i in range(len(matrix)):
        f.write("\"" + str(matrix[i]) + "\"")
        f.write(f",{scores[i]},{spec_scores[i]}\n")

    f.close()


def perform_dimensionality_reduction(trajectories_file, save_file):
    matrix = []

    trajectory_data = pd.read_csv(trajectories_file)
    count = 0

    for index, row in trajectory_data.iterrows():
        matrix.append([int(el) for el in row["Trajectory"].strip("\"[]").split(",")])

        '''count += 1
        if count >= 1000:
            break'''

    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(np.array(matrix))
    np.save(save_file, pca_result)


def perform_clustering(trajectories_file, save_file_prefix):
    matrix = []

    trajectory_data = pd.read_csv(trajectories_file)
    count = 0

    for index, row in trajectory_data.iterrows():
        matrix.append([int(el) for el in row["Trajectory"].strip("\"[]").split(",")])

        count += 1
        #if count >= 1000:
        #    break

    # No DTW
    results = KMeans(n_clusters=4, random_state=0).fit(matrix)

    # DTW
    #results = TimeSeriesKMeans(n_clusters=4, metric="dtw", n_init=1, max_iter=10, random_state=0).fit(matrix)

    labels = results.labels_
    centroids = results.cluster_centers_

    np.save(f"{save_file_prefix}_labels.npy", labels)
    np.save(f"{save_file_prefix}_centroids.npy", centroids)


def generate_cluster_plots(data_file_prefix, save_file_prefix):
    labels = np.load(f"{data_file_prefix}_labels.npy")
    centroids = np.load(f"{data_file_prefix}_centroids.npy")
    centroid_names = ["Novice", "Dropper", "Generalist", "Collector"]

    # Cluster centroids
    '''fig, axs = plt.subplots(2, 2, sharex=True, sharey=True)
    x = np.linspace(0, centroids.shape[1], centroids.shape[1])

    for i in range(centroids.shape[0]):
        axs[i // 2][i % 2].plot(x,centroids[i],)
        axs[i // 2][i % 2].title.set_text(centroid_names[i])

    plt.suptitle("Observed Agent Phenotype Centroids")
    plt.xlabel("Simulation Time Steps")
    plt.ylabel("Y Location of Agent")
    plt.savefig(f"{save_file_prefix}_centroids.pdf")

    fig = plt.figure()
    strategy_counts = [0]*len(centroid_names)
    for i in range(len(labels)):
        strategy_counts[labels[i]] += 1

    plt.bar(centroid_names, strategy_counts)
    plt.xlabel("Phenotype")
    plt.ylabel("Number of Episodes")
    plt.title("Occurrences of Each Phenotype")
    plt.savefig(f"{save_file_prefix}_counts.pdf")
    print(strategy_counts)'''

    for i in range(len(centroids)):
        for j in range(i, len(centroids)):
            dist = np.linalg.norm(centroids[i] - centroids[j])
            print(f"{centroid_names[i]}->{centroid_names[j]} = {dist}")





def generate_3D_plot(reduction_file, trajectories_file, save_file):
    scores = []
    spec_scores = []
    trajectory_data = pd.read_csv(trajectories_file)

    count=0

    for index, row in trajectory_data.iterrows():
        scores.append(row[" Fitness"])
        spec_scores.append(row[" Specialisation"])

        '''count += 1
        if count >= 1000:
            break'''

    pca_result = np.load(reduction_file)
    x = pca_result[:10000, 0]
    y = pca_result[:10000, 1]
    z = scores[:10000]
    c = spec_scores[:10000]

    fig = plt.figure()

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
    ax = fig.add_subplot(projection='3d')
    ax.scatter3D(x, y, z, c=c, vmin=0, vmax=1, cmap=cm)
    plt.title("3D Mapping of Phenotype Landscape")
    # plt.savefig(plot_destination)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Trajectory Clustering')
    parser.add_argument('--generate_trajectories', action="store")
    parser.add_argument('--rwg_genomes_file', action="store")
    parser.add_argument('--parameter_path', action="store")
    parser.add_argument('--save_file', action="store")
    parser.add_argument('--perform_reduction', action="store")
    parser.add_argument('--trajectories_file', action="store")
    parser.add_argument('--plot_3D', action="store")
    parser.add_argument('--reduction_file', action="store")
    parser.add_argument('--clustering', action="store")
    parser.add_argument('--save_file_prefix', action="store")
    parser.add_argument('--plot_centroids', action='store')
    parser.add_argument('--data_file_prefix', action="store")

    generate_trajectories = parser.parse_args().generate_trajectories
    generate_trajectories = True if generate_trajectories and generate_trajectories.lower() == "true" else False
    rwg_genomes_file = parser.parse_args().rwg_genomes_file
    parameter_path = parser.parse_args().parameter_path
    save_file = parser.parse_args().save_file

    perform_reduction = parser.parse_args().perform_reduction
    perform_reduction = True if perform_reduction and perform_reduction.lower() == "true" else False
    trajectories_file = parser.parse_args().trajectories_file

    clustering = parser.parse_args().clustering
    clustering = True if clustering and clustering.lower() == "true" else False
    save_file_prefix = parser.parse_args().save_file_prefix

    plot_3D = parser.parse_args().plot_3D
    plot_3D = True if plot_3D and plot_3D.lower() == "true" else False
    reduction_file = parser.parse_args().reduction_file

    plot_centroids = parser.parse_args().plot_centroids
    plot_centroids = True if plot_centroids and plot_centroids == "true" else False
    data_file_prefix = parser.parse_args().data_file_prefix

    if generate_trajectories:
        generate_trajectory_data(rwg_genomes_file, parameter_path, save_file)
    elif perform_reduction:
        perform_dimensionality_reduction(trajectories_file, save_file)
    elif clustering:
        perform_clustering(trajectories_file, save_file_prefix)
    elif plot_3D:
        generate_3D_plot(reduction_file, trajectories_file, save_file)
    elif plot_centroids:
        generate_cluster_plots(data_file_prefix, save_file_prefix)


# Do PCA on new data

# Plot