import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import matplotlib.colors as colours
from matplotlib.colors import ListedColormap

from scipy import stats
from operator import add

setups = ["Fully-centralised", "Centralised", "Decentralised"]
setup_labels = ["Fully Centralised", "CTDE", "Fully Decentralised"]
alpha = 0.2
background_alpha = 0.5
spec_metric_index = 0 # R_coop
#spec_metric_index = 2 # R_spec
#spec_metric_index = 5 # R_spec_P


def from_string(arr_str):
    """
    Convert string to a 2D array
    @param arr_str:
    @return:
    """
    arr_str = arr_str[1:-1]
    a = []
    list_to_parse = arr_str.strip('').split("]")[:-1]

    for el in list_to_parse:
        new_el = el.replace("[","")
        new_el = [float(item) for item in new_el.split()]
        a += [new_el]

    return a


def plot_episodes(path_to_results, path_to_graph, min_agents, max_agents, y_height=15000, exclude_middle_plots=False):
    # Prepare data lists
    x = [i for i in range(2, max_agents + 2, 2)]
    y_centralised = []
    y_decentralised = []
    y_onepop = []
    y_homogeneous = []
    y_fully_centralised = []

    # Read data from results file
    data = pd.read_csv(path_to_results)

    results = {}

    for setup in setups:
        for team_size in range(min_agents, max_agents + 2, 2):
            key = f"{setup}-{team_size}"
            results[key] = {}

    for index, row in data.iterrows():
        learning_type = row["learning_type"].capitalize()
        team_type = row["team_type"].capitalize()
        reward_level = row["reward_level"].capitalize()
        num_agents = row["num_agents"]
        seed = row["seed"]

        if num_agents < min_agents or num_agents > max_agents:
            continue

        if learning_type == "Centralised" and reward_level == "Individual":
            key = f"One-pop-{num_agents}"
        elif team_type == "Homogeneous":
            key = f"{team_type}-{num_agents}"
        else:
            key = f"{learning_type}-{num_agents}"

        if seed not in results[key]:
            results[key][seed] = None

        results[key][seed] = {"fitness": from_string(row["fitness"]),
                              "specialisation": from_string(row["specialisation"])}

    if exclude_middle_plots and min_agents!=max_agents:
        fig = plt.figure(figsize=(9.5, 9))
    elif exclude_middle_plots:
        #fig = plt.figure(figsize=(5, 9))
        fig = plt.figure(figsize=(18, 4))
    else:
        fig = plt.figure(figsize=(19, 9))

    vermillion = (213, 94, 0)
    bluish_green = (0, 158, 115)

    generalist_colour = vermillion
    specialist_colour = bluish_green

    # Divide RGB value by 256 to map it to 0-1 range
    N = 256
    vals = np.ones((N, 4))
    vals[:, 0] = np.linspace(generalist_colour[0] / N, specialist_colour[0]/N, N)
    vals[:, 1] = np.linspace(generalist_colour[1] / N, specialist_colour[1]/N, N)
    vals[:, 2] = np.linspace(generalist_colour[2] / N, specialist_colour[2]/N, N)

    # Create custom colour map
    cm = ListedColormap(vals)

    #cm = plt.cm.get_cmap('RdYlGn')
    norm = colours.Normalize(vmin=0, vmax=1)
    m = plt.cm.ScalarMappable(norm=norm, cmap=cm)

    suptitle_font = 18
    title_font = 16
    label_font = 16
    tick_font = 15
    legend_font = 12
    suptitle_padding = 1.0
    title_padding = 1.02

    for row in range(1, len(setups)+1):
        if exclude_middle_plots and min_agents != max_agents:
            col_max = 3
        elif exclude_middle_plots:
            col_max = 2

        else:
            col_max = max_agents//2 + 1

        col_range = range(1, col_max)

        for col in col_range:
            #plot = ((row - 1) * (col_max-1)) + col
            plot = ((row - 1) * (col_max - 1)) + col
            plot = ((col - 1) * (len(setups))) + row

            setup = setups[row-1]
            #ax = fig.add_subplot(len(setups), col_max - 1, plot)#, sharey=True)#, sharex=True)
            ax = fig.add_subplot(col_max - 1, len(setups),  plot)

            if exclude_middle_plots and col == 2:
                num_agents = max_agents
            elif exclude_middle_plots:
                num_agents = (min_agents // 2 + (col - 1)) * 2
            else:
                num_agents = col*2

            key = f"{setup}-{num_agents}"
            #plot_name = f"{setup_labels[row-1]}-{num_agents} agents"
            plot_name = f"{setup_labels[row - 1]}"

            all_runs = [None]*len(results[key])

            for index, seed in enumerate(results[key]):

                team_fitness_list = [0] * len(results[key][seed]["fitness"][0])
                specialisation_list = results[key][seed]["specialisation"]

                for j in range(num_agents):
                    team_fitness_list = list(map(add, team_fitness_list, results[key][seed]["fitness"][j]))

                these_episodes = [None] * len(team_fitness_list)

                for episode_index, episode_fitness in enumerate(team_fitness_list):
                    colour = m.to_rgba(specialisation_list[episode_index][spec_metric_index])
                    #ax.plot(index, episode_fitness/num_agents, 'o', color=colour, alpha=alpha, markersize=3)
                    these_episodes[episode_index] = (episode_fitness/num_agents, colour)

                #ax.plot(index, np.mean(team_fitness_list)/num_agents, 'x', color='black')
                all_runs[index] = (np.mean(team_fitness_list)/num_agents, these_episodes)

            all_runs = sorted(all_runs, key=lambda x: x[0])

            for index,run in enumerate(all_runs):
                ax.plot(index, run[0], 'x', color='black', alpha=background_alpha)

                for episode in run[1]:
                    ax.plot(index, episode[0], 'o', color=episode[1], alpha=alpha, markersize=3)

            mean_all_runs = np.mean([all_runs[i][0] for i in range(len(all_runs))])
            ax.plot([x for x in range(len(all_runs))], [mean_all_runs]*len(all_runs), color='black')

            ax.set_ylim(-2000, y_height)

            #if col==1:

            plt.setp(ax.get_xticklabels(), fontsize=tick_font)

            #if row==(len(setups)):


            #else:
            #    ax.set_xticklabels([])

            if row != 1:
                ax.set_yticklabels([])

            else:
                ax.set_ylabel("Fitness per Agent", fontsize=label_font, x=title_padding)

            ax.set_xlabel("Evolutionary Run Number", fontsize=label_font)

            plt.setp(ax.get_yticklabels(), fontsize=tick_font)

            ax.set_title(plot_name, fontsize=title_font, y=title_padding)

    #fig.colorbar(m)
    plt.tight_layout(pad=1.5)
    #plt.suptitle("Performance Across Episodes", fontsize=suptitle_font, y=suptitle_padding)
    plt.savefig(path_to_graph)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot episodes')
    parser.add_argument('--results_path', action="store")
    parser.add_argument('--graph_path', action="store")
    parser.add_argument('--min_agents', action="store")
    parser.add_argument('--max_agents', action="store")
    parser.add_argument('--y_height', action="store")
    parser.add_argument('--exclude_middle_plots', action="store")

    results_path = parser.parse_args().results_path
    graph_path = parser.parse_args().graph_path
    min_agents = int(parser.parse_args().min_agents)
    max_agents = int(parser.parse_args().max_agents)
    y_height = parser.parse_args().y_height
    exclude_middle_plots = parser.parse_args().exclude_middle_plots

    if y_height:
        y_height = int(y_height)

    if exclude_middle_plots == "True":
        exclude_middle_plots = True
    elif exclude_middle_plots == "False":
        exclude_middle_plots = False
    else:
        raise RuntimeError("Add parameter for excluding (or not excluding) middle plots")

    plot_episodes(results_path, graph_path, min_agents, max_agents, y_height, exclude_middle_plots)