import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse

from scipy import stats
from operator import add
from evaluation.plot_scalability import from_string
from sklearn.decomposition import PCA

# WARNING: The ith element of each list must match
# i.e. setup_labels[i] must be the shortened name for setups[i] etc
setups = ["Centralised", "Decentralised", "Fully-centralised"]
setup_labels = ["CTDE", "Fully Decentralised", "Fully Centralised"]
marker_list = ['^', 's', 'o']
setup_markers = {
                "Centralised": '^',
                "Decentralised": 's',
                "Fully-centralised": 'o'
                }

spec_metric_index = 2  # R_spec


def plot_behaviours(path_to_results, path_to_graph, max_agents, bc_measure):
    # Read data from results file
    data = pd.read_csv(path_to_results)

    results = {}

    for setup in setups:
        for team_size in range(2, max_agents + 2, 2):
            key = f"{setup}-{team_size}"
            results[key] = {}

    for index, row in data.iterrows():
        learning_type = row["learning_type"].capitalize()
        team_type = row["team_type"].capitalize()
        reward_level = row["reward_level"].capitalize()
        num_agents = row["num_agents"]
        seed = row["seed"]

        if learning_type == "Centralised" and reward_level == "Individual":
            key = f"One-pop-{num_agents}"
        elif team_type == "Homogeneous":
            key = f"{team_type}-{num_agents}"
        else:
            key = f"{learning_type}-{num_agents}"

        if seed not in results[key]:
            results[key][seed] = None

        if type(row["fitness"]) == float:
            results[key][seed] = {"fitness": row["fitness"]
                                  ,"specialisation": row["specialisation"],
                                  "behaviour_characterisation": row["behaviour_characterisation"]}
        else:
            results[key][seed] = {"fitness": from_string(row["fitness"]),
                                  "specialisation": from_string(row["specialisation"]),
                                  "behaviour_characterisation": from_string(row["behaviour_characterisation"], dim=3)}

    fig = plt.figure(figsize=(19, 9))
    cm = plt.cm.get_cmap('RdYlGn')
    plt.suptitle("Mapping Evolved Solutions in the Behaviour Space")
    num_rows = 3
    num_cols = max_agents // 2

    for num_agents in range(2, max_agents+2, 2):
        matrix = []
        markers = []
        labels = []
        scores = []
        setup_start_id = [None for _ in range(len(setups))]
        current_matrix_id = 0

        for i in range(len(setups)):
            setup = setups[i]
            key = f"{setup}-{num_agents}"
            setup_start_id[i] = current_matrix_id

            for seed in results[key]:
                if bc_measure == "agent_action_count":
                    num_episodes = len(results[key][seed]["behaviour_characterisation"][0])

                    for episode in range(num_episodes):
                        team_bc = [0 for _ in range(len(results[key][seed]["behaviour_characterisation"][0][0]))]

                        for j in range(num_agents):
                            team_bc = list(map(add, team_bc, results[key][seed]["behaviour_characterisation"][j][episode]))

                        matrix += [team_bc]
                        current_matrix_id += 1
                        markers += [setup_markers[setup]]
                        labels += [setup]
                        specialisation = results[key][seed]["specialisation"][episode][spec_metric_index]
                        #specialisation_each_episode = [episode[spec_metric_index] for episode in results[key][seed]["specialisation"]]
                        #specialisation = np.mean(specialisation_each_episode)
                        scores += [specialisation]

                else:
                    team_bc = []

                    for agent_bc in results[key][seed]["behaviour_characterisation"]:
                            team_bc += agent_bc

                    results[key][seed]["behaviour_characterisation"] = team_bc
                    matrix += [team_bc]
                    markers += [setup_markers[setup]]
                    labels += [setup]

                    specialisation_each_episode = [episode[spec_metric_index] for episode in results[key][seed]["specialisation"]]
                    specialisation = np.mean(specialisation_each_episode)
                    scores += [specialisation]

        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(np.array(matrix))
        print(f"{num_agents}: {np.sum(pca.explained_variance_ratio_)}")

        for setup_id in range(len(setups)):
            start_id = setup_start_id[setup_id]

            if setup_id < len(setups) - 1:
                end_id = setup_start_id[setup_id + 1]
                xs = pca_result[start_id:end_id, 0]
                ys = pca_result[start_id:end_id, 1]
                scores_temp = scores[start_id:end_id]
                markers_temp = markers[start_id:end_id]
                labels_temp = labels[start_id:end_id]

            else:
                xs = pca_result[start_id:, 0]
                ys = pca_result[start_id:, 1]
                scores_temp = scores[start_id:]
                markers_temp = markers[start_id:]
                labels_temp = labels[start_id:]

            subplot_id = (num_agents//2) + (setup_id * num_cols)
            ax = fig.add_subplot(num_rows, num_cols, subplot_id)
            plt.title(f"{setup_labels[setup_id]}- {num_agents} Agents")

            for x, y, s, m, l in zip(xs, ys, scores_temp, markers_temp, labels_temp):
                ax.scatter(x, y, c=cm(s), marker=m, label=l)

    #fig.legend(setup_labels)
    plt.savefig(path_to_graph)




path_to_results = "/Users/mostafa/Documents/Code/PhD/TS-Platform/results/2021_08_06_a_scalability_variable_arena/data/results_reevaluated_30ep_final.csv"
path_to_graph = "/Users/mostafa/Documents/Code/PhD/TS-Platform/results/2021_08_06_a_scalability_variable_arena/analysis/behaviour_plot.png"
plot_behaviours(path_to_results, path_to_graph, max_agents=8, bc_measure="agent_action_count")