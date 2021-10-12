import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse

from scipy import stats
from operator import add

setups = ["Centralised", "Decentralised", "Fully-centralised"]
setup_labels = ["CTDE", "FD", "FC"]

ctde_colour = '#d55e00'
decentralised_colour = '#0071b2'
fully_centralised_colour = '#009E73'

spec_metric_index = 0 # R_coop
#spec_metric_index = 2 # R_spec
#spec_metric_index = 5 # R_spec_P
#spec_metric_index = 0

spec_index_with_participation = spec_metric_index + 3
spec_index_without_participation = spec_metric_index


def from_string(arr_str, dim=2):
    """
    Convert string to an array with dimension dim
    @param arr_str:
    @param dim: Dimension size of new array
    @return:
    """
    if dim == 1:
        arr_str = arr_str[1:-1]
        a = []
        list_to_parse = arr_str.split()

        for el in list_to_parse:
            a += [float(el)]

        return a

    if dim == 2:
        arr_str = arr_str[1:-1]
        a = []
        list_to_parse = arr_str.strip('').split("]")[:-1]

        for el in list_to_parse:
            new_el = el.replace("[","")
            new_el = [float(item) for item in new_el.split()]
            a += [new_el]

        return a

    if dim == 3:
        arr_str = arr_str[1:-1]
        a = []
        list_to_parse = arr_str.strip('').split("]]")[:-1]

        for el in list_to_parse:
            new_el = from_string(el+"]]")
            a += [new_el]

        return a





def plot_scalability(path_to_results, path_to_graph, plot_type, max_agents, violin=False, y_min_height=-2000, y_height=15000, showing="fitness"):
    # Prepare data lists
    x = [i for i in range(2, max_agents+2, 2)]
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
            results[key][seed] = {"fitness": row["fitness"],
                                  "specialisation": row["specialisation"],
                                  "participation": row["participation"]}
        else:
            results[key][seed] = {"fitness": from_string(row["fitness"]),
                                  "specialisation": from_string(row["specialisation"]),
                                  "participation": from_string(row["participation"])}

    scores = {}

    for setup in setups:
        for team_size in range(2, max_agents + 2, 2):
            key = f"{setup}-{team_size}"
            scores[key] = []

    for num_agents in range(2, max_agents+2, 2):
        for setup in setups:
            key = f"{setup}-{num_agents}"

            for seed in results[key]:
                if showing == "fitness":
                    team_fitness_list = [0] * len(results[key][seed]["fitness"][0])

                    for j in range(num_agents):
                        team_fitness_list = list(map(add, team_fitness_list, results[key][seed]["fitness"][j]))

                    scores[key] += [np.mean(team_fitness_list)/num_agents]

                elif showing == "specialisation":
                    specialisation_each_episode = [episode[spec_metric_index] for episode in results[key][seed]["specialisation"]]
                    specialisation = np.mean(specialisation_each_episode)
                    scores[key] += [specialisation]

                elif showing == "participation":
                    participation_each_episode = []

                    for episode in results[key][seed]["participation"]:
                        agents_participated = 0

                        for agent_res_count in episode:
                            if agent_res_count != 0:
                                agents_participated += 1

                        participation_each_episode += [agents_participated / num_agents]

                    participation = np.mean(participation_each_episode)
                    scores[key] += [participation]


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
                y += [np.mean(scores[key])]
                #yerr += [np.std(scores[key])]
            elif plot_type == "best":
                y += [np.max(scores[key])]
                yerr += [stats.sem(scores[key])]
            elif plot_type == "median":
                y += [np.median(scores[key])]
                yerr += [stats.sem(scores[key])]
            elif plot_type == "no_variance":
                y += [np.mean(scores[key])]
            elif plot_type == "error":
                y += [np.mean(scores[key])]
                yerr += [stats.sem(scores[key])]

    for key in results:
        print(f"{key}: {len(results[key])}")

    # Plot
    title_font = 40
    label_font = 36
    tick_font = 30
    legend_font = 30
    label_padding = 1.08

    ''''''
    if not violin:
        fig1, ax1 = plt.subplots(figsize=(16, 10))
        #ax1.set_title(f'Scalability of {showing.capitalize()} with Number of Agents', fontsize=title_font, y=label_padding)

        if showing == "fitness":
            ax1.set_ylim(0, y_height)
            ax1.set_ylabel('Fitness per agent', fontsize=label_font, x=label_padding)

        elif showing == "specialisation":
            ax1.set_ylim(0, 1.1)
            ax1.set_ylabel('Team Specialisation', fontsize=label_font)

        elif showing == "participation":
            ax1.set_ylim(0, 1.1)
            ax1.set_ylabel('Team Participation', fontsize=label_font)

        ax1.set_xticks([x for x in range(2, max_agents+2, 2)])
        ax1.set_xlabel('Number of Agents', fontsize=label_font, y=label_padding)

        for tick in ax1.xaxis.get_major_ticks():
            tick.label.set_fontsize(tick_font)

        for tick in ax1.yaxis.get_major_ticks():
            tick.label.set_fontsize(tick_font)

        linewidth = 5
        markersize = 17

        if plot_type == "error":
            plt.errorbar(x, y_centralised, yerr_centralised, color=ctde_colour, linestyle='-', label="CTDE", marker='^', markersize=markersize, linewidth=linewidth)
            plt.errorbar(x, y_decentralised, yerr_decentralised, color=decentralised_colour, linestyle='-', label="Fully Decentralised", marker='s', markersize=markersize, linewidth=linewidth)
            plt.errorbar(x, y_fully_centralised, yerr_fully_centralised, color=fully_centralised_colour, linestyle='-', label="Fully Centralised", marker='o', markersize=markersize, linewidth=linewidth)

        elif plot_type == "mean":
            plt.plot(x, y_centralised, color=ctde_colour, linestyle='-', label="CTDE", marker='^', markersize=markersize, linewidth=linewidth)
            plt.plot(x, y_decentralised, color=decentralised_colour, linestyle='-', label="Fully Decentralised", marker='s', markersize=markersize, linewidth=linewidth)
            plt.plot(x, y_fully_centralised, color=fully_centralised_colour, linestyle='-', label="Fully Centralised", marker='o', markersize=markersize, linewidth=linewidth)


        elif plot_type == "best" or plot_type == "median":
            plt.plot(x, y_centralised, 'ro-', label=f"CTDE ({plot_type})")
            plt.plot(x, y_decentralised, 'bo-', label=f"Fully Decentralised ({plot_type})")
            plt.plot(x, y_fully_centralised, 'go-', label=f"Fully Centralised ({plot_type})")
            # plt.plot(x, y_onepop, 'go-', label=f"One-pop ({plot_type})")
            # plt.plot(x, y_homogeneous, 'ko-', label=f"Homogeneous ({plot_type})")

        else:
            plt.plot(x, y_centralised, 'ro-', label="CTDE")
            plt.plot(x, y_decentralised, 'bo-', label="Fully Decentralised")
            #plt.plot(x, y_onepop, 'go-', label="One-pop")
            #plt.plot(x, y_homogeneous, 'ko-', label="Homogeneous")
            plt.plot(x, y_fully_centralised, 'go-', label="Fully Centralised")

        plt.legend(fontsize=legend_font)# loc='upper right',
        plt.savefig(path_to_graph)

    else:
        def set_axis_style(ax, labels):
            ax.get_xaxis().set_tick_params(direction='out')
            ax.xaxis.set_ticks_position('bottom')
            ax.set_xticks(np.arange(1, len(labels) + 1))
            ax.set_xticklabels(labels, fontsize=tick_font)
            ax.set_xlim(0.25, len(labels) + 0.75)
            for tick in ax.yaxis.get_major_ticks():
                tick.label.set_fontsize(tick_font)

        #fig = plt.figure(figsize=(19, 9))
        fig, axs = plt.subplots(1, 4, sharey=True, figsize=(19,9))
        num_cols = 3

        for col,ax in enumerate(axs):

            if showing == "fitness":
                ax.set_ylim(y_min_height, y_height)
                ax.set_ylabel("Fitness per Agent", fontsize=label_font)
            elif showing == "specialisation" or showing == "participation":
                ax.set_ylim(-0.2, 1.2)
                ax.set_ylabel("Team Specialisation", fontsize=label_font)

            num_agents = (col+1) * 2
            parts = ax.violinplot([scores[f"{setup}-{num_agents}"] for setup in setups])

            for id,pc in enumerate(parts['bodies']):
                if setups[id] == "Centralised":
                    pc.set_color(ctde_colour)
                elif setups[id] == "Decentralised":
                    pc.set_color(decentralised_colour)
                elif setups[id] == "Fully-centralised":
                    pc.set_color(fully_centralised_colour)
                #elif setups[id] == "Homogeneous":
                #    pc.set_color('black')

            for pc in ('cbars', 'cmins', 'cmaxes'):
                parts[pc].set_color('black')

            set_axis_style(ax, setup_labels)
            ax.set_title(f"{num_agents} Agents", fontsize=label_font)

        plt.suptitle(f"{showing.capitalize()} Distribution", fontsize=title_font)
        plt.savefig(path_to_graph)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot scalability')
    parser.add_argument('--results_path', action="store")
    parser.add_argument('--graph_path', action="store")
    parser.add_argument('--plot_type', action="store")
    parser.add_argument('--max_agents', action="store")
    parser.add_argument('--violin', action="store")
    parser.add_argument('--y_height_min', action="store")
    parser.add_argument('--y_height_max', action="store")
    parser.add_argument('--showing', action="store")

    results_path = parser.parse_args().results_path
    graph_path = parser.parse_args().graph_path
    plot_type = parser.parse_args().plot_type
    max_agents = int(parser.parse_args().max_agents)
    violin = parser.parse_args().violin

    if not violin or violin != "True":
        violin = False
    else:
        violin = True

    y_height_min = parser.parse_args().y_height_min
    y_height_max = parser.parse_args().y_height_max

    if y_height_min:
        y_height_min = int(y_height_min)

    if y_height_max:
        y_height_max = int(y_height_max)

    showing = parser.parse_args().showing

    plot_scalability(results_path, graph_path, plot_type, max_agents, violin, y_height_min, y_height_max, showing)