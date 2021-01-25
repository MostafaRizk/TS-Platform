import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from helpers import sammon
from mpl_toolkits import mplot3d

rwg_genomes_file = "../sammon_testing_rwg.csv"
f = open(rwg_genomes_file, "r")
rwg_data = f.read().strip().split("\n")
N_episodes = 20
min_score = -20000
max_score = 200000
matrix = []
scores = []
rwg_indices = [i for i in range(10)] + [i for i in range(-10,0)]

evolved_genomes_directory = "/Users/mostafa/Documents/Code/PhD/TS-Platform/results/2020_11_24_magic_plot_combined_new_seed/results"

specialisation_file = "specialisation_final.csv"
specialisation_data = pd.read_csv(specialisation_file)
spec_score_keys = ["R_coop", "R_coop_eff", "R_spec", "R_coop x P", "R_coop_eff x P", "R_spec x P"]
spec_scores = {}
for key in spec_score_keys:
    spec_scores[key] = []

num_rwg = 0
num_team = 0
num_ind = 0

# Get rwg data
for index in rwg_indices:
    row = rwg_data[index]
    genome = [float(element) for element in row.split(",")[0:-N_episodes]]
    episode_scores = [float(score) for score in row.split(",")[-N_episodes:]]
    mean_score = np.mean(episode_scores)
    matrix += [genome]
    scores += [mean_score]

    # Find row in spec dataframe with matching index and filename
    matching_rows = specialisation_data.loc[ (specialisation_data['Model Name'] == str(index)) & (specialisation_data['Model Directory'] == rwg_genomes_file) ]

    if len(matching_rows) != 1:
        raise RuntimeError("Matching entries in the specialisation file is not 1")

    # Store spec scores from the dataframe in the dictionary
    for index, row in matching_rows.iterrows():
        for key in spec_scores.keys():
            spec_scores[key] += [row[key]]


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
            # Add fitness to scores
            scores += [row["fitness"]]

            num_team += 1

        ''''''
        # Duplicate genome if it has an individual reward
        if row["reward_level"] == "individual" and row["num_agents"] == 2:
            # Get genome from corresponding model file
            model_file = os.path.join(evolved_genomes_directory, row["model_name"])
            genome = np.load(model_file)
            # Add genome to matrix
            matrix += [np.append(genome,genome)]
            # Add fitness to scores
            scores += [row["fitness"]*2]
            num_ind += 1

        if row["num_agents"] == 2:
            # Find row in spec dataframe with matching index and filename
            matching_rows = specialisation_data.loc[(specialisation_data['Model Name'] == row["model_name"]) & (specialisation_data['Model Directory'] == evolved_genomes_directory)]

            if len(matching_rows) != 1:
                raise RuntimeError("Matching entries in the specialisation file is not 1")

            # Store spec scores from the dataframe in the dictionary
            for index2, row2 in matching_rows.iterrows():
                for key in spec_scores.keys():
                    spec_scores[key] += [row2[key]]

scores = np.array(scores)

# By default, sammon returns a 2-dim array and the error E
[new_data, error] = sammon.sammon(np.array(matrix), n=2)
x = new_data[:, 0]
y = new_data[:, 1]
z = scores

rwg_start_index = 0
team_start_index = 0 + num_rwg
ind_start_index = team_start_index + num_team

fig = plt.figure(figsize=(19, 9))

for i in range(1,7):
    ax = fig.add_subplot(2, 3, i, projection='3d')
    cm = plt.cm.get_cmap('RdYlGn')
    key = spec_score_keys[i-1]

    #sc = plt.scatter(x, y, c=scores, vmin=min_score, vmax=max_score, cmap=cm)
    p = ax.scatter3D(x[rwg_start_index : team_start_index], y[rwg_start_index : team_start_index], z[rwg_start_index : team_start_index], c=spec_scores[key][rwg_start_index : team_start_index], vmin=0, vmax=1, cmap=cm, label="Rwg genomes")
    p = ax.scatter3D(x[team_start_index : ind_start_index], y[team_start_index : ind_start_index], z[team_start_index : ind_start_index], c=spec_scores[key][team_start_index : ind_start_index], vmin=0, vmax=1, cmap=cm, label="Team Genomes")
    #ax.scatter3D(x[ind_start_index:], y[ind_start_index:], z[ind_start_index:], marker='X', c=scores[ind_start_index:], vmin=min_score, vmax=max_score, cmap=cm, label="Individual Genomes")
    fig.colorbar(p)
    plt.title(key)

#ax.colorbar()
plt.suptitle("3D Mapping of Fitness Landscape")
#fig.legend(loc="lower left")
plt.savefig("sammon_3D.png")
#plt.show()


