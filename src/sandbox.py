# Get all decentralised*2agents*final.npy files and put in list according to seeds
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import math
from glob import glob
from fitness import FitnessCalculator
from agents.nn_agent_lean import NNAgent
from sklearn.cluster import KMeans

from evaluation.evaluate_model import evaluate_model

parent_directory = "/Users/mostafa/Documents/Code/PhD/TS-Platform/results/2022_09_14_replicator_mutator/"
model_directory = os.path.join(parent_directory, "decentralised_data")
cluster_file_prefix = os.path.join(parent_directory, "kmeans")


def generate_files():
    # Get list of all npy files using glob
    all_genome_files = glob(f"{model_directory}/*final.npy")
    all_parameter_files = glob(f"{model_directory}/*json")
    std = 0.01
    num_samples = 30
    num_episodes = 5
    num_agents = 2

    # Load each model's npy data into a dictionary with seeds as key
    model_dict = {}
    paramfile_dict = {}

    for filename in all_genome_files:
        seed = filename.split('/')[-1].split('_')[6]

        if seed not in model_dict:
            model_dict[seed] = []

        model_dict[seed].append(np.load(filename))

    for filename in all_parameter_files:
        seed = filename.split('/')[-1].split('_')[6]

        paramfile_dict[seed] = filename

    # Convert each npy array into a distribution
    samples = {}
    for seed in model_dict.keys():
        num_dimensions = model_dict[seed][0].shape[0]
        covariance = [[0 for _ in range(num_dimensions)] for _ in range(num_dimensions)]
        for i in range(num_dimensions):
            covariance[i][i] = std**2
        assert len(model_dict[seed]) == num_agents, f"The number of agents with this seed is not equal to {num_agents}"
        samples[seed] = []

        for i in range(num_agents):
            samples[seed].append(np.random.multivariate_normal(model_dict[seed][i], covariance, num_samples))

    # Evaluate the samples and log the trajectories for each seed
    trajectories = {}
    for seed in samples.keys():
        trajectories[seed] = []
        parameter_path = paramfile_dict[seed]
        fitness_calculator = FitnessCalculator(parameter_path)
        for j in range(len(samples[seed][0])):
            genomes_list = [samples[seed][i][j] for i in range(num_agents)]
            agent_list = [NNAgent(fitness_calculator.get_observation_size(), fitness_calculator.get_action_size(), parameter_path, genomes_list[i]) for i in range(len(genomes_list))]
            results = fitness_calculator.calculate_fitness(controller_list=agent_list, render=False, measure_specialisation=True, logging=False, logfilename=None, render_mode="human")

            for i in range(num_agents):
                for ep in range(num_episodes):
                    trajectories[seed].append(results["trajectory_matrix"][i][ep])
                    #scores.append(results["fitness_matrix"][i][ep])
                    #spec_scores.append(results["specialisation_list"][ep][chosen_key])

    f = open("/Users/mostafa/Documents/Code/PhD/TS-Platform/results/2022_09_14_replicator_mutator/cma_trajectories_variance.csv", "w")
    f.write("Seed,Trajectories\n")

    for seed in trajectories.keys():
        f.write(f"{seed},")
        f.write("\"" + str(trajectories[seed]) + "\"" + "\n")
    f.close()

    # Find distance to each centroid
    labels = np.load(f"{cluster_file_prefix}_labels.npy")
    centroids = np.load(f"{cluster_file_prefix}_centroids.npy")
    centroid_names = ["Novice", "Dropper", "Generalist", "Collector"]

    population = {}

    for seed in trajectories.keys():
        population[seed] = {strategy: 0 for strategy in centroid_names}
        total = 0
        for trajectory in trajectories[seed]:
            min_dist = float('inf')
            nearest_centroid = None
            for i in range(len(centroids)):
                dist = np.linalg.norm(centroids[i] - trajectory)
                if dist < min_dist:
                    min_dist = dist
                    nearest_centroid = centroid_names[i]
            population[seed][nearest_centroid] += 1
            total += 1

        for key in population[seed].keys():
            population[seed][key] /= total

    f = open("/Users/mostafa/Documents/Code/PhD/TS-Platform/results/2022_09_14_replicator_mutator/cma_population_variance.csv", "w")
    f.write("Seed,Novice,Dropper,Generalist,Collector\n")

    for seed in population.keys():
        #print(seed)
        #print(population[seed])
        f.write(str(seed) + "," + str(population[seed]['Novice']) + "," + str(population[seed]['Dropper']) + "," + str(population[seed]['Generalist']) + "," + str(population[seed]['Collector']) + "\n")
    f.close()


def perform_clustering(trajectories_file, save_file_prefix, num_strategies):
    matrix = []

    trajectory_data = pd.read_csv(trajectories_file)
    count = 0

    for index, row in trajectory_data.iterrows():
        trajectories = row["Trajectories"].split("]")[:-2]
        for traj in trajectories:
            split_list = traj.strip(", ")
            split_list = split_list.replace("\"", "")
            split_list = split_list.replace("[", "")
            split_list = split_list.replace("]", "")
            split_list = split_list.replace(" ", "")
            split_list = split_list.split(",")
            matrix.append([int(el) for el in split_list])

        count += 1
        #if count >= 1000:
        #    break

    results = KMeans(n_clusters=num_strategies, random_state=0).fit(matrix)

    labels = results.labels_
    centroids = results.cluster_centers_

    np.save(f"{save_file_prefix}_k={num_strategies}_labels_cma_v.npy", labels)
    np.save(f"{save_file_prefix}_k={num_strategies}_centroids_cma_v.npy", centroids)

    f = open(f"{save_file_prefix}_k={num_strategies}_error_v.txt", "w")
    f.write(str(results.inertia_))
    f.close()


def plot_cluster_data(trajectories_file, data_file_prefix, num_strategies):
    labels = np.load(f"{data_file_prefix}_k={num_strategies}_labels_cma_v.npy")
    centroids = np.load(f"{data_file_prefix}_k={num_strategies}_centroids_cma_v.npy")
    #centroid_names = ["Collector", "Dropper", "Generalist"]
    centroid_names = ["Generalist", "Collector", "Dropper", "Generalist"]

    fig, axs = plt.subplots(math.ceil(num_strategies/2), 2, sharex=True, sharey=True)
    # fig.set_figheight(15)
    # fig.set_figwidth(5)
    x = np.linspace(0, centroids.shape[1], centroids.shape[1])

    for i in range(centroids.shape[0]):
        axs[i // 2][i % 2].plot(x,centroids[i],)
        axs[i // 2][i % 2].set_title(centroid_names[i], fontsize=18, pad=5)
        plt.setp(axs[i // 2][i % 2].get_xticklabels(), fontsize=12)
        plt.setp(axs[i // 2][i % 2].get_yticklabels(), fontsize=12)

    #plt.show()

    plt.suptitle("Observed Agent Phenotype Centroids", fontsize=18)
    #plt.xlabel("Simulation Time Steps")
    #plt.ylabel("Y Location of Agent")
    fig.text(0.5, 0.01, "Simulation Time Steps", ha='center', fontsize=15)
    fig.text(0.04, 0.5, "Y Location of Agent", va='center', rotation='vertical', fontsize=15)
    plt.savefig(f"{data_file_prefix}_cma_centroids_k={num_strategies}_v.pdf")

    '''fig = plt.figure()
    strategy_counts = [0]*len(centroid_names)
    for i in range(len(labels)):
        strategy_counts[labels[i]] += 1

    plt.bar(centroid_names, strategy_counts)
    plt.xlabel("Phenotype")
    plt.ylabel("Number of Episodes")
    plt.title("Occurrences of Each Phenotype")
    plt.savefig(f"{data_file_prefix}_cma_counts_k={num_strategies}_v.pdf")'''

    if num_strategies == 3:
        population = {}
        trajectory_data = pd.read_csv(trajectories_file)

        for index, row in trajectory_data.iterrows():
            trajectories = row["Trajectories"].split("]")[:-2]
            population[row["Seed"]] = {strategy: 0 for strategy in centroid_names}
            total = 0

            for traj in trajectories:
                split_list = traj.strip(", ")
                split_list = split_list.replace("\"", "")
                split_list = split_list.replace("[", "")
                split_list = split_list.replace("]", "")
                split_list = split_list.replace(" ", "")
                split_list = split_list.split(",")
                traj_refined = [int(el) for el in split_list]

                min_dist = float('inf')
                nearest_centroid = None
                for i in range(len(centroids)):
                    dist = np.linalg.norm(centroids[i] - traj_refined)
                    if dist < min_dist:
                        min_dist = dist
                        nearest_centroid = centroid_names[i]
                population[row["Seed"]][nearest_centroid] += 1
                total += 1

            for key in population[row["Seed"]].keys():
                population[row["Seed"]][key] /= total

        f = open("/Users/mostafa/Documents/Code/PhD/TS-Platform/results/2022_09_14_replicator_mutator/cma_population_new_v.csv", "w")
        f.write("Seed,Collector,Dropper,Generalist\n")

        for seed in population.keys():
            f.write(str(seed) + "," + str(population[seed]['Collector']) + "," + str(population[seed]['Dropper']) + "," + str(population[seed]['Generalist']) + "\n")
        f.close()

#generate_files()
trajectories_file = os.path.join(parent_directory, "cma_trajectories_variance.csv")
#perform_clustering(trajectories_file, cluster_file_prefix, 4)
#perform_clustering(trajectories_file, cluster_file_prefix, 3)

#plot_cluster_data(trajectories_file, cluster_file_prefix, 3)
plot_cluster_data(trajectories_file, cluster_file_prefix, 4)

# Visualise arena
#model_path = "/Users/mostafa/Documents/Code/PhD/TS-Platform/results/2022_09_14_replicator_mutator/decentralised_data/decentralised_cma_heterogeneous_individual_nn_slope_4282876140_2_4_1_8_8_4_1_3_7_1_3.0_0.2_2_1000_100_5_False_rnn_False_1_4_tanh_100_0.2_1000_0.001_0.0_1000_0.0_1_6836.839999999998_final.npy"
#evaluate_model(model_path=model_path, episodes=1, rendering=True, time_delay=0, print_scores=False, ids_to_remove=None)