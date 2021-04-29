import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import math

from evaluation.plot_scalability import from_string
from scipy import stats
from operator import add

setups = ["Centralised", "Decentralised", "One-pop", "Homogeneous"]
spec_metric_index = 2 # R_spec
#spec_metric_index = 5 # R_spec_P


def plot_robustness(data_path, file_list, path_to_graph, plot_type, max_agents, y_height=15000, showing="fitness"):
    # Create results dictionary with keys for num_agents_removed
    results = {}

    for setup in setups:
        for team_size in range(2, max_agents+2, 2):
            key = f"{setup}-{team_size}"
            results[key] = {}

            for num_agents_removed in range(len(file_list)):
                if num_agents_removed < team_size:
                    results[key][num_agents_removed] = {}

    # For each results file, populate dictionary
    for num_agents_removed, results_file in enumerate(file_list):
        path_to_results = os.path.join(data_path, results_file)
        data = pd.read_csv(path_to_results)

        for index, row in data.iterrows():
            learning_type = row["learning_type"].capitalize()
            team_type = row["team_type"].capitalize()
            reward_level = row["reward_level"].capitalize()
            num_agents = row["num_agents"]

            if num_agents > max_agents:
                continue

            seed = row["seed"]
            ids_removed = row["agents_removed"]

            if not ids_removed:
                ids_removed = 0

            if learning_type == "Centralised" and reward_level == "Individual":
                key = f"One-pop-{num_agents}"
            elif team_type == "Homogeneous":
                key = f"{team_type}-{num_agents}"
            else:
                key = f"{learning_type}-{num_agents}"

            if seed not in results[key][num_agents_removed]:
                results[key][num_agents_removed][seed] = {}

            results[key][num_agents_removed][seed][ids_removed] = {"fitness": from_string(row["fitness"]),
                                                                   "specialisation": from_string(row["specialisation"])}

    # Create figure with 2 subplots per row
    cols = 2
    rows = math.ceil((max_agents / 2) / cols)
    fig, axs = plt.subplots(rows, cols, sharey=True, figsize=(30, 15))
    fig.suptitle(f"Robustness of {showing.capitalize()}")

    scores = {}

    for setup in setups:
        for team_size in range(2, max_agents + 2, 2):
            key = f"{setup}-{team_size}"
            scores[key] = {}

            for num_agents_removed in range(len(file_list)):
                if num_agents_removed < team_size:
                    scores[key][num_agents_removed] = []

    # For each team size
    for i in range((max_agents)//2):
        num_agents = (i+1) * 2
        col_id = i % cols

        if rows > 1:
            row_id = i // rows
            axis = axs[row_id][col_id]

        else:
            axis = axs[col_id]

        x = [j for j in range(num_agents)]
        y_centralised = []
        y_decentralised = []
        y_onepop = []
        y_homogeneous = []

        yerr_centralised = []
        yerr_decentralised = []
        yerr_onepop = []
        yerr_homogeneous = []

        # For each number of agents removed (that is valid for that team size)
        for num_agents_removed in range(num_agents):
            for setup in setups:
                key = f"{setup}-{num_agents}"

                # For ever seed
                for seed in results[key][num_agents_removed]:

                    # For every combination of removed ids
                    scores_for_combination = []

                    for ids_removed in results[key][num_agents_removed][seed]:
                        if showing == "fitness":
                            team_fitness_list = [0] * len(results[key][num_agents_removed][seed][ids_removed]["fitness"][0])

                            for j in range(num_agents):
                                team_fitness_list = list(map(add, team_fitness_list, results[key][num_agents_removed][seed][ids_removed]["fitness"][j]))

                            scores_for_combination += [np.mean(team_fitness_list) / num_agents]

                        elif showing == "specialisation":
                            specialisation_each_episode = [episode[spec_metric_index] for episode in results[key][num_agents_removed][seed][ids_removed]["specialisation"]]
                            specialisation = np.mean(specialisation_each_episode)
                            scores_for_combination += [specialisation]

                    scores[key][num_agents_removed] += [np.mean(scores_for_combination)]

                if setup == "Centralised":
                    y = y_centralised
                    yerr = yerr_centralised

                elif setup == "Decentralised":
                    y = y_decentralised
                    yerr = yerr_decentralised

                elif setup == "One-pop":
                    y = y_onepop
                    yerr = yerr_onepop

                elif setup == "Homogeneous":
                    y = y_homogeneous
                    yerr = yerr_homogeneous

                if plot_type == "mean":
                    y += [np.mean(scores[key][num_agents_removed])]
                    yerr += [np.std(scores[key][num_agents_removed])]
                elif plot_type == "best":
                    y += [np.max(scores[key][num_agents_removed])]
                    yerr += [stats.sem(scores[key][num_agents_removed])]
                elif plot_type == "median":
                    y += [np.median(scores[key][num_agents_removed])]
                    yerr += [stats.sem(scores[key][num_agents_removed])]
                elif plot_type == "no_variance":
                    y += [np.mean(scores[key][num_agents_removed])]
                elif plot_type == "error":
                    y += [np.mean(scores[key][num_agents_removed])]
                    yerr += [stats.sem(scores[key][num_agents_removed])]

        for key in results:
            print(f"{key}: {len(results[key])}")

        # Add data to plot
        axis.label_outer()
        axis.set_title(f"{num_agents} Agents", fontsize=20)
        axis.set_xlabel("Num Agents Removed", fontsize=18)

        for tick in axis.xaxis.get_major_ticks():
            tick.label.set_fontsize(16)

        for tick in axis.yaxis.get_major_ticks():
            tick.label.set_fontsize(16)

        if showing == "fitness":
            axis.set_ylabel("Fitness per agent", fontsize=18)
            axis.set_ylim(0, y_height)

        elif showing == "specialisation":
            axis.set_ylabel("Specialisation", fontsize=18)
            axis.set_ylim(0, 1.1)

        if plot_type == "mean" or plot_type == "error":
            axis.errorbar(x, y_centralised, yerr_centralised, fmt='r-', label="Centralised")
            axis.errorbar(x, y_decentralised, yerr_decentralised, fmt='b-', label="Decentralised")
            axis.errorbar(x, y_onepop, yerr_onepop, fmt='g-', label="One-pop")
            axis.errorbar(x, y_homogeneous, yerr_homogeneous, fmt='k-', label="Homogeneous")

        elif plot_type == "best" or plot_type == "median":
            axis.plot(x, y_centralised, 'ro-', label=f"Centralised ({plot_type})")
            axis.plot(x, y_decentralised, 'bo-', label=f"Decentralised ({plot_type})")
            axis.plot(x, y_onepop, 'go-', label=f"One-pop ({plot_type})")
            axis.plot(x, y_homogeneous, 'ko-', label=f"Homogeneous ({plot_type})")

        else:
            axis.plot(x, y_centralised, 'ro-', label="Centralised")
            axis.plot(x, y_decentralised, 'bo-', label="Decentralised")
            axis.plot(x, y_onepop, 'go-', label="One-pop")
            axis.plot(x, y_homogeneous, 'ko-', label="Homogeneous")

        axis.legend(loc='upper right', fontsize=16)

    # Save
    #filename = f"robustness_of_{showing}.png"
    plt.savefig(path_to_graph)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot scalability')
    parser.add_argument('--data_path', action="store")
    parser.add_argument('--file_list', action="store")
    parser.add_argument('--graph_path', action="store")
    parser.add_argument('--plot_type', action="store")
    parser.add_argument('--max_agents', action="store")
    parser.add_argument('--y_height', action="store")
    parser.add_argument('--showing', action="store")

    data_path = parser.parse_args().data_path
    file_list = parser.parse_args().file_list

    file_list = file_list.strip("[]").split(",")

    graph_path = parser.parse_args().graph_path
    plot_type = parser.parse_args().plot_type
    max_agents = int(parser.parse_args().max_agents)

    y_height = parser.parse_args().y_height

    if y_height:
        y_height = int(y_height)

    showing = parser.parse_args().showing

    plot_robustness(data_path, file_list, graph_path, plot_type, max_agents, y_height, showing)