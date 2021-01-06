from helpers import sammon
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

genomes_file = "/Users/mostafa/Documents/Code/PhD/TS-Platform/results/2020_09_04_Identical_oller_comparison/results/all_genomes_rwg_heterogeneous_team_nn_slope_1_2_4_1_4_8_4_1_3_7_1_3.0_0.2_2_1000_500_20_rnn_False_1_4_tanh_20000_normal_0_1_.csv"
f = open(genomes_file, "r")
data = f.read().strip().split("\n")
N_episodes = 20
matrix = []
scores = []

for row in data:
    genome = [float(element) for element in row.split(",")[0:-N_episodes]]
    episode_scores = [float(score) for score in row.split(",")[-N_episodes:]]
    mean_score = np.mean(episode_scores)
    matrix += [genome]
    scores += [mean_score]

scores = np.array(scores)
interpolated_scores = np.interp(scores, (-20000, 250000), (0,1))
areas = (30 * interpolated_scores)**2  # 0 to 15 point radii


# By default, sammon returns a 2-dim array and the error E
[new_data, error] = sammon.sammon(np.array(matrix), n=2)
x = new_data[:, 0]
y = new_data[:, 1]
plt.scatter(x, y, interpolated_scores)
plt.savefig("sammon_1.png")
