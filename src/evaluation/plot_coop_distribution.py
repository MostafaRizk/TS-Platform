import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import matplotlib.colors as colours
from matplotlib.colors import ListedColormap
from matplotlib.ticker import PercentFormatter
from matplotlib.patches import Rectangle

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


def plot_coop_distribution(path_to_results, path_to_graph, agents=None, averaged=True):
    # Read data from results file
    data = pd.read_csv(path_to_results)

    results = {}

    if agents:
        max_agents = agents + 2

    else:
        agents = 2
        max_agents = 10

    for setup in setups:
        for team_size in range(agents, max_agents, 2):
            key = f"{setup}-{team_size}"
            results[key] = {}

    for index, row in data.iterrows():
        learning_type = row["learning_type"].capitalize()
        team_type = row["team_type"].capitalize()
        reward_level = row["reward_level"].capitalize()
        num_agents = row["num_agents"]
        seed = row["seed"]

        if num_agents != agents:
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

    fig = plt.figure(figsize=(19, 9))

    vermillion = (213, 94, 0)
    bluish_green = (0, 158, 115)

    generalist_colour = vermillion
    specialist_colour = bluish_green

    # Divide RGB value by 256 to map it to 0-1 range
    N = 256

    generalist_colour = tuple([val/N for val in list(generalist_colour)])
    specialist_colour = tuple([val/N for val in list(specialist_colour)])

    suptitle_font = 18
    title_font = 26
    label_font = 26
    tick_font = 15
    legend_font = 12
    suptitle_padding = 1.0
    title_padding = 1.02

    key = f"Decentralised-{agents}"
    plot_name = f"Fully-Decentralised"
    list_of_coop_scores = []

    for index, seed in enumerate(results[key]):
        specialisation_list = results[key][seed]["specialisation"]

        coop_scores_this_run = []

        for episode_index in range(len(specialisation_list)):
            coop_scores_this_run += [specialisation_list[episode_index][spec_metric_index]]

        if averaged:
            list_of_coop_scores += [np.mean(coop_scores_this_run)]

        else:
            list_of_coop_scores += coop_scores_this_run

    all_bins = True

    if all_bins:
        n, bins, patches = plt.hist(list_of_coop_scores, 10, weights=np.ones(len(list_of_coop_scores)) / len(list_of_coop_scores))

        print(len(bins))
        print(len(patches))
        for i in range(len(patches)//2):
            patches[i].set_facecolor(generalist_colour)
        for i in range(len(patches)//2, len(patches)):
            patches[i].set_facecolor(specialist_colour)
    else:
        n, bins, patches = plt.hist(list_of_coop_scores, 2, weights=np.ones(len(list_of_coop_scores)) / len(list_of_coop_scores))

        patches[0].set_facecolor(generalist_colour)
        patches[1].set_facecolor(specialist_colour)

    handles = [Rectangle((0, 0), 1, 1, color=c, ec="k") for c in [generalist_colour, specialist_colour]]
    labels = ["Selfish Episodes", "Cooperative Episodes"]
    plt.legend(handles, labels, fontsize=label_font)

    print(n)

    plt.xlabel('Degree of Cooperation', fontsize=label_font)
    plt.ylabel('Number of Episodes', fontsize=label_font)
    plt.xticks(bins, fontsize=tick_font)
    plt.yticks(fontsize=tick_font)
    plt.title(f'Distribution of Cooperation Scores Across Episodes For {agents}-Agent Teams', fontsize=title_font)
    #plt.xlim(0.0, 1.0)
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    plt.savefig(path_to_graph)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot episodes')
    parser.add_argument('--results_path', action="store")
    parser.add_argument('--graph_path', action="store")
    parser.add_argument('--agents', action="store")
    parser.add_argument('--averaged', action="store")

    results_path = parser.parse_args().results_path
    graph_path = parser.parse_args().graph_path
    agents = int(parser.parse_args().agents)
    averaged = parser.parse_args().averaged

    if averaged == "True":
        averaged = True
    elif averaged == "False":
        averaged = False
    else:
        raise RuntimeError("Invalid value for parameter named: averaged")

    plot_coop_distribution(results_path, graph_path, agents, averaged)