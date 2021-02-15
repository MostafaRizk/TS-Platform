import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from helpers import sammon
from mpl_toolkits import mplot3d
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

#rwg_genomes_files = ["rwg_with_spec.csv"] #+ [f"extra_rwg_{i}.csv" for i in range(6,7)]
#rwg_genomes_files = [f"extra_rwg_{i}.csv" for i in range(3,4)]
#rwg_genomes_files = ["rwg_lhs_5.csv"]
rwg_genomes_files = ["lhs_new_spec.csv"]

rwg_data = []
for rwg_genomes_file in rwg_genomes_files:
    f = open(rwg_genomes_file, "r")
    rwg_data += f.read().strip().split("\n")
N_episodes = 20
min_score = -20000
max_score = 200000
matrix = []
scores = []
rwg_indices = [i for i in range(60)] + [i for i in range(-60,0)]

evolved_genomes_directory = "/Users/mostafa/Documents/Code/PhD/TS-Platform/results/2020_11_24_magic_plot_combined_new_seed/results"

specialisation_file = "data/plots/specialisation_equal_samples.csv"
specialisation_data = pd.read_csv(specialisation_file)
spec_score_keys = ["R_coop", "R_coop_eff", "R_spec", "R_coop x P", "R_coop_eff x P", "R_spec x P"]
spec_scores = {}
for key in spec_score_keys:
    spec_scores[key] = []

num_rwg = 0
num_team = 0
num_ind = 0

# Number of scores (fitness and specialisation)
num_spec_scores = len(spec_score_keys)*N_episodes
num_scores = N_episodes + num_spec_scores


# Get rwg data
for index in rwg_indices:
    row = rwg_data[index]
    genome = [float(element) for element in row.split(",")[0:-num_scores]]
    episode_scores = [float(score) for score in row.split(",")[-num_scores:-num_spec_scores]]

    # Make a list of lists. For each specialisation metric, there is a list of scores for all episodes
    raw_spec_scores = row.split(",")[-num_spec_scores:]
    raw_spec_scores = [[float(raw_spec_scores[i+j]) for i in range(0, len(raw_spec_scores), len(spec_score_keys))] for j in range(len(spec_score_keys))]

    '''
    max_episode_index = None
    max_score = float('-Inf')

    for i in range(len(episode_scores)):
        if episode_scores[i] > max_score:
            max_score = episode_scores[i]
            max_episode_index = i

    # Take the max of all episodes for each metric and store them in the dictionary
    
    for i in range(len(spec_score_keys)):
        spec_scores[spec_score_keys[i]] += [raw_spec_scores[i][max_episode_index]]
    
    scores += [max_score]
    '''

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
    results_file = os.path.join(evolved_genomes_directory,f"results_{generation}.csv")
    # Load all genomes into pandas dataframe
    evolved_data = pd.read_csv(results_file)
    # For each genome
    for index, row in evolved_data.iterrows():
        # Check if it has team reward and 2 agents
        if row["reward_level"] == "team" and row["num_agents"] == 2:
            # Get genome from corresponding model file
            model_file = os.path.join(evolved_genomes_directory, row["model_name"])
            genome = np.load(model_file)
            # Add genome to matrix
            matrix += [genome]

            num_team += 1

        ''''''
        # Duplicate genome if it has an individual reward
        if row["reward_level"] == "individual" and row["num_agents"] == 2:
            # Get genome from corresponding model file
            model_file = os.path.join(evolved_genomes_directory, row["model_name"])
            genome = np.load(model_file)
            # Add genome to matrix
            matrix += [np.append(genome,genome)]

            num_ind += 1

        if row["num_agents"] == 2:
            # Find row in spec dataframe with matching index and filename
            matching_rows = specialisation_data.loc[(specialisation_data['Model Name'] == row["model_name"]) & (specialisation_data['Model Directory'] == evolved_genomes_directory)]

            if len(matching_rows) != 1:
                raise RuntimeError("Matching entries in the specialisation file is not 1")

            # Store spec scores from the dataframe in the dictionary
            for index2, row2 in matching_rows.iterrows():
                # Add fitness to scores
                scores += [row2["Team Fitness"]]

                # Add specialisation scores
                for key in spec_scores.keys():
                    spec_scores[key] += [row2[key]]

scores = np.array(scores)

# Do dimensionality reduction using PCA
'''
pca = PCA(n_components=200)
pca_result = pca.fit_transform(np.array(matrix))
print(np.sum(pca.explained_variance_ratio_))
x = pca_result[:,0]
y = pca_result[:,1]
'''

# Do dimensionality reduction using t-SNE
''''''
tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=1000)
tsne_results = tsne.fit_transform(np.array(matrix))
#tsne_results = tsne.fit_transform(pca_result)
x = tsne_results[:,0]
y = tsne_results[:,1]


z = scores

rwg_start_index = 0
team_start_index = 0 + num_rwg
ind_start_index = team_start_index + num_team

fig = plt.figure(figsize=(19, 9))

num_cols = 3

for row in range(1, 4):
    for col in range(1, num_cols+1):
        plot = ((row-1) * num_cols) + col

        ax = fig.add_subplot(3, 3, plot, projection='3d')
        cm = plt.cm.get_cmap('RdYlGn')
        key = spec_score_keys[col-1]
        plot_name = f"RWG ({key})"

        p = ax.scatter3D(x[rwg_start_index : team_start_index], y[rwg_start_index : team_start_index], z[rwg_start_index : team_start_index], c=spec_scores[key][rwg_start_index : team_start_index], vmin=0, vmax=1, cmap=cm, label="Rwg genomes")

        if row == 2 or row == 3:
            p = ax.scatter3D(x[team_start_index : ind_start_index], y[team_start_index : ind_start_index], z[team_start_index : ind_start_index], c=spec_scores[key][team_start_index : ind_start_index], vmin=0, vmax=1, cmap=cm, label="Team Genomes")
            plot_name = f"RWG + Teams ({key})"

        if row == 3:
            p = ax.scatter3D(x[ind_start_index:], y[ind_start_index:], z[ind_start_index:], c=spec_scores[key][ind_start_index:], vmin=0, vmax=1, cmap=cm, label="Individual Genomes")
            plot_name = f"RWG + Teams + Individuals ({key})"

        fig.colorbar(p)
        plt.title(plot_name)

#ax.colorbar()
plt.suptitle("3D Mapping of Fitness Landscape")
#fig.legend(loc="lower left")
#plt.savefig(f"tsne_new_spec_lhs_5_60_samples.png")
plt.savefig(f"tsne_lhs_all_variants.png")
#plt.show()


