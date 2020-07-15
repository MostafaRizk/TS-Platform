from scripts.benchmarking import BenchmarkPlotter
import numpy as np

# Best mean score so far
best_scores = []

# Mean score for each sample (genome)
all_scores = []

# List of scores in all episodes (simulation runs) for each sample (genome)
# 50,000 lists of length 5, in my case
all_trials = []

# Weights of
best_weights = []

L0_weights = []
L1_weights = []
L2_weights = []

N_samples = 50000
N_episodes = 5
total_runtime = -1.0

#####
rwg_genomes_file = "all_genomes_rwg_homogeneous_team_nn_1_2_3_1_4_8_4_1_3_7_1_3.0_0.2_2_1000_500_5_rnn_True_[]_linear_100000_normal_0_1_.csv"
f = open(rwg_genomes_file, "r")
data = f.read().strip().split("\n")

best_weights = np.array([float(element) for element in data[0].split(",")[0:-N_episodes]])
best_score = np.mean([float(episode_score) for episode_score in data[0].split(",")[-N_episodes:]])

for row in data:
    genome = np.array([float(element) for element in row.split(",")[0:-N_episodes]])
    episode_scores = [float(score) for score in row.split(",")[-N_episodes:]]
    mean_score = np.mean(episode_scores)
    all_scores += [mean_score]

    if mean_score > best_score:
        best_score = mean_score
        best_weights = genome

    best_scores += [best_score]
    all_trials += [episode_scores]

    L0 = sum([np.sum(w) for w in genome]) / len(genome)
    L1 = sum([np.abs(w).sum() for w in genome]) / len(genome)
    L2 = sum([(w ** 2).sum() for w in genome]) / len(genome)

    L0_weights.append(L0)
    L1_weights.append(L1)
    L2_weights.append(L2)


#####

data_dict = {
            'best_scores' : best_scores,
            'all_scores' : all_scores,
            'all_trials' : all_trials,
            'best_weights' : [bw.tolist() for bw in best_weights],
            'L0_weights' : L0_weights,
            'L1_weights' : L1_weights,
            'L2_weights' : L2_weights,
            'N_samples' : N_samples,
            'N_episodes' : N_episodes,
            'total_runtime' : total_runtime
        }

plotter = BenchmarkPlotter()
plotter.save_all_sample_stats(data_dict)