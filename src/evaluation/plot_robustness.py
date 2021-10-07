import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import math
import chardet

from evaluation.plot_scalability import from_string
from scipy import stats
from operator import add

setups = ["Centralised", "Decentralised", "Fully-centralised"]
setup_labels = ["CTDE", "FD", "FC"]
spec_metric_index = 0 # R_coop
# spec_metric_index = 2 # R_spec
# spec_metric_index = 5 # R_spec_P

ctde_colour = '#d55e00'
decentralised_colour = '#0071b2'
fully_centralised_colour = '#009E73'


def plot_robustness(data_path, file_list, path_to_graph, plot_type, min_agents, max_agents, agents_removed, y_min_height, y_height, showing="fitness"):
    # Create results dictionary with keys for num_agents_removed
    results = {}

    for setup in setups:
        for team_size in range(min_agents, max_agents+2, 2):
            key = f"{setup}-{team_size}"
            results[key] = {}

            for num_agents_removed in range(len(file_list)):
                if num_agents_removed < team_size:
                    results[key][num_agents_removed] = {}

    # For each results file, populate dictionary
    for num_agents_removed, results_file in enumerate(file_list):
        path_to_results = os.path.join(data_path, results_file)

        with open(path_to_results, 'rb') as rawdata:
            file_attributes = chardet.detect(rawdata.read(100000))

        data = pd.read_csv(path_to_results, encoding=file_attributes['encoding'])

        for index, row in data.iterrows():
            learning_type = row["learning_type"].capitalize()
            team_type = row["team_type"].capitalize()
            reward_level = row["reward_level"].capitalize()
            num_agents = row["num_agents"]

            if num_agents > max_agents or num_agents < min_agents:
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
    rows = 1
    #cols = math.ceil((max_agents / 2) / rows)
    cols = (max_agents//2 - min_agents//2) + 1

    # Plot
    suptitle_font = 60
    title_font = 55
    label_font = 50
    tick_font = 45
    legend_font = 45
    label_padding = 1.08

    fig, axs = plt.subplots(rows, cols, sharey=True, figsize=(30, 15))
    #fig.suptitle(f"Robustness of {showing.capitalize()}", fontsize=suptitle_font)

    scores = {}

    for setup in setups:
        for team_size in range(min_agents, max_agents + 2, 2):
            key = f"{setup}-{team_size}"
            scores[key] = {}

            for num_agents_removed in range(len(file_list)):
                if num_agents_removed < team_size:
                    scores[key][num_agents_removed] = []

    # For each team size
    for i in range(min_agents, max_agents+2, 2):
        #num_agents = (i+1) * 2
        num_agents = i
        col_id = (i - min_agents)//2 + 1

        if cols > 1:
            axis = axs[col_id]

        else:
            axis = axs

        x = [j for j in range(agents_removed+1)]
        y_centralised = []
        y_decentralised = []
        y_onepop = []
        y_homogeneous = []
        y_fully_centralised = []

        yerr_centralised = []
        yerr_decentralised = []
        yerr_onepop = []
        yerr_homogeneous = []
        yerr_fully_centralised = []

        if num_agents < min_agents or num_agents > max_agents:
            continue

        # For each number of agents removed (that is valid for that team size)
        for num_agents_removed in range(agents_removed+1):
            for setup in setups:
                key = f"{setup}-{num_agents}"

                # For every seed
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

                elif setup == "Fully-centralised":
                    y = y_fully_centralised
                    yerr = yerr_fully_centralised

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
        #axis.set_title(f"{num_agents} Agents", fontsize=title_font)
        axis.set_title(f"Robustness of {showing.capitalize()} for {num_agents} Agents", fontsize=suptitle_font, y=label_padding)
        axis.set_xlabel("Num Agents Removed", fontsize=title_font)
        axis.set_xticks(np.arange(0, agents_removed + 1))
        axis.set_xticklabels([str(k) for k in range(agents_removed+1)])

        for tick in axis.xaxis.get_major_ticks():
            tick.label.set_fontsize(tick_font)

        for tick in axis.yaxis.get_major_ticks():
            tick.label.set_fontsize(tick_font)

        if showing == "fitness":
            axis.set_ylabel("Fitness per agent", fontsize=label_font)
            axis.set_ylim(y_min_height, y_height)

        elif showing == "specialisation":
            axis.set_ylabel("Specialisation", fontsize=label_font)
            axis.set_ylim(0, 1.1)

        linewidth = 8
        markersize = 26

        if plot_type == "error":
            axis.errorbar(x, y_centralised, yerr_centralised, color=ctde_colour, linestyle='-', label="CTDE", marker='^', markersize=markersize, linewidth=linewidth)
            axis.errorbar(x, y_decentralised, yerr_decentralised, color=decentralised_colour, linestyle='-', label="Fully Decentralised", marker='s', markersize=markersize, linewidth=linewidth)
            axis.errorbar(x, y_fully_centralised, yerr_fully_centralised, color=fully_centralised_colour, linestyle='-', label="Fully Centralised", marker='o', markersize=markersize, linewidth=linewidth)

        elif plot_type == "mean":
            axis.plot(x, y_centralised, color=ctde_colour, linestyle='-', label="CTDE", marker='^', markersize=markersize, linewidth=linewidth)
            axis.plot(x, y_decentralised, color=decentralised_colour, linestyle='-', label="Fully Decentralised", marker='s', markersize=markersize, linewidth=linewidth)
            axis.plot(x, y_fully_centralised, color=fully_centralised_colour, linestyle='-', label="Fully Centralised", marker='o', markersize=markersize, linewidth=linewidth)

        else:
            axis.plot(x, y_centralised, 'ro-', label="CTDE")
            axis.plot(x, y_decentralised, 'bo-', label="Fully-Decentralised")
            #axis.plot(x, y_onepop, 'go-', label="One-pop")
            #axis.plot(x, y_homogeneous, 'ko-', label="Homogeneous")
            axis.plot(x, y_fully_centralised, 'go-', label="Fully-Centralised")

        axis.legend(loc='upper right', fontsize=legend_font)

    # Save
    #filename = f"robustness_of_{showing}.png"
    plt.savefig(path_to_graph)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot scalability')
    parser.add_argument('--data_path', action="store")
    parser.add_argument('--file_list', action="store")
    parser.add_argument('--graph_path', action="store")
    parser.add_argument('--plot_type', action="store")
    parser.add_argument('--min_agents', action="store")
    parser.add_argument('--max_agents', action="store")
    parser.add_argument('--agents_removed', action="store")
    parser.add_argument('--y_min_height', action="store")
    parser.add_argument('--y_height', action="store")
    parser.add_argument('--showing', action="store")

    data_path = parser.parse_args().data_path
    file_list = parser.parse_args().file_list

    file_list = file_list.strip("[]").split(",")

    graph_path = parser.parse_args().graph_path
    plot_type = parser.parse_args().plot_type
    min_agents = int(parser.parse_args().min_agents)
    max_agents = int(parser.parse_args().max_agents)
    agents_removed = int(parser.parse_args().agents_removed)

    y_min_height = parser.parse_args().y_min_height

    if y_min_height:
        y_min_height = int(y_min_height)

    y_height = parser.parse_args().y_height

    if y_height:
        y_height = int(y_height)

    showing = parser.parse_args().showing

    plot_robustness(data_path, file_list, graph_path, plot_type, min_agents, max_agents, agents_removed, y_min_height, y_height, showing)