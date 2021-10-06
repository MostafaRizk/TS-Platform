import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from helpers import sammon
from mpl_toolkits import mplot3d
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from evaluation.plot_scalability import from_string
from operator import add
from glob import glob

sliding_speed = 8
num_agents = 2
chosen_key = 2 # Chosen spec metric

rwg_genomes_files = [
    f"/Users/mostafa/Documents/Code/PhD/TS-Platform/results/2021_10_05_rwg_only_ctde/data/all_genomes_centralised_rwg_heterogeneous_team_nn_slope_1791095846_2_4_1_8_8_4_1_3_7_1_3.0_0.2_2_1000_100_20_False_rnn_False_1_4_tanh_10000_normal_0_1_sorted.csv"]

rwg_data = []

for rwg_genomes_file in rwg_genomes_files:
    f = open(rwg_genomes_file, "r")
    rwg_data += f.read().strip().split("\n")

N_episodes = 20
min_score = -2000
max_score = 200000
matrix = []
scores = []
#rwg_indices = [i for i in range(-1000,0)]
#rwg_indices = [i for i in range(15)] + [i for i in range(-15,0)]
rwg_indices = [i for i in range(-100,0)]

evolved_genomes_directory = "/Users/mostafa/Documents/Code/PhD/TS-Platform/results/2021_08_06_a_scalability_variable_arena/data"

'''
specialisation_file = f"/Users/mostafa/Documents/Code/PhD/TS-Platform/src/scripts/experiments/different_slopes/specialisation_different_slopes_{sliding_speed}.csv"
specialisation_data = pd.read_csv(specialisation_file)
'''
spec_score_keys = ["R_coop", "R_coop_eff", "R_spec", "R_coop x P", "R_coop_eff x P", "R_spec x P"]
spec_scores = {}
for key in spec_score_keys:
    spec_scores[key] = []

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
for generation in ["final"]:
    results_file = os.path.join(evolved_genomes_directory,f"results_reevaluated_30ep_final.csv")
    # Load all genomes into pandas dataframe
    evolved_data = pd.read_csv(results_file)
    # For each genome
    for index, row in evolved_data.iterrows():
        # Check if it has team reward and 2 agents
        if row["num_agents"] == num_agents and row["sliding_speed"] == sliding_speed:
            '''if row["reward_level"] == "team" and row["learning_type"] == "centralised":
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
            '''

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
tsne = TSNE(n_components=2, verbose=1)
tsne_results = tsne.fit_transform(np.array(matrix))
# tsne_results = tsne.fit_transform(pca_result)
x = tsne_results[:, 0]
y = tsne_results[:, 1]

z = scores

# rwg_start_index = 0
# team_start_index = 0 + num_rwg
# ind_start_index = team_start_index + num_team

# fig = plt.figure(figsize=(19, 9))

# num_cols = 3



fig = plt.figure()
cm = plt.cm.get_cmap('RdYlGn')
key = spec_score_keys[chosen_key]
ax = fig.add_subplot(projection='3d')
ax.scatter3D(x, y, z, c=spec_scores[key], vmin=0, vmax=1, cmap=cm)
# ax.colorbar()
plt.title("3D Mapping of Fitness Landscape")
# fig.legend(loc="lower left")
# plt.savefig(f"tsne_new_spec_lhs_5_60_samples.png")
# plt.savefig(f"dimensionality_reduction.png")
plt.show()


