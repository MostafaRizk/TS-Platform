from helpers import sammon
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

rwg_genomes_file = "../sammon_testing_rwg.csv"
f = open(rwg_genomes_file, "r")
rwg_data = f.read().strip().split("\n")
N_episodes = 20
min_score = -20000
max_score = 200000
matrix = []
scores = []
colours = []
rwg_data_segment = rwg_data[:500] + rwg_data[-500:]

evolved_genomes_directory = "/Users/mostafa/Documents/Code/PhD/TS-Platform/results/2020_11_24_magic_plot_combined_new_seed/results"
results_file = os.path.join(evolved_genomes_directory,"results_final.csv")
# Load all genomes into pandas dataframe
evolved_data = pd.read_csv(results_file)

num_rwg = 0
num_team = 0
num_ind = 0

# Get rwg data
for row in rwg_data_segment:
    genome = [float(element) for element in row.split(",")[0:-N_episodes]]
    episode_scores = [float(score) for score in row.split(",")[-N_episodes:]]
    mean_score = np.mean(episode_scores)
    matrix += [genome]
    scores += [mean_score]
    colours += ['b']
    num_rwg += 1

# Get evolved data
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
        colours += ['r']
        num_team += 1

    # Duplicate genome if it has an individual reward
    if row["reward_level"] == "individual" and row["num_agents"] == 2:
        # Get genome from corresponding model file
        model_file = os.path.join(evolved_genomes_directory, row["model_name"])
        genome = np.load(model_file)
        # Add genome to matrix
        matrix += [np.append(genome,genome)]
        # Add fitness to scores
        scores += [row["fitness"]*2]
        colours += ['g']
        num_ind += 1

scores = np.array(scores)
interpolated_scores = np.interp(scores, (min_score, max_score), (0,1))
areas = (30 * interpolated_scores)**2  # 0 to 15 point radii


# By default, sammon returns a 2-dim array and the error E
[new_data, error] = sammon.sammon(np.array(matrix), n=2)
x = new_data[:, 0]
y = new_data[:, 1]
#plt.scatter(x, y, areas, colours)
#plt.savefig("sammon.png")
#plt.show()

rwg_start_index = 0
team_start_index = 0 + num_rwg
ind_start_index = team_start_index + num_team

cm = plt.cm.get_cmap('RdYlGn')
#sc = plt.scatter(x, y, c=scores, vmin=min_score, vmax=max_score, cmap=cm)
plt.scatter(x[rwg_start_index : team_start_index], y[rwg_start_index : team_start_index], marker='.', c=scores[rwg_start_index : team_start_index], vmin=min_score, vmax=max_score, cmap=cm, label="Rwg genomes")
plt.scatter(x[team_start_index : ind_start_index], y[team_start_index : ind_start_index], marker='o', c=scores[team_start_index : ind_start_index], vmin=min_score, vmax=max_score, cmap=cm, label="Team Genomes")
plt.scatter(x[ind_start_index:], y[ind_start_index:], marker='X', c=scores[ind_start_index:], vmin=min_score, vmax=max_score, cmap=cm, label="Individual Genomes")
plt.colorbar()
plt.title("2D Mapping of Fitness Landscape")
plt.legend(loc="lower left")
plt.savefig("sammon.png")


