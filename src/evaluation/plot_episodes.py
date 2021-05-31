import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import matplotlib.colors as colours

from scipy import stats
from operator import add

setups = ["Centralised", "Decentralised", "One-pop", "Homogeneous"]
alpha = 0.2
background_alpha = 0.5
#spec_metric_index = 2 # R_spec
spec_metric_index = 5 # R_spec_P


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


def plot_episodes(path_to_results, path_to_graph, max_agents, y_height=15000):
    # Prepare data lists
    x = [i for i in range(2, max_agents + 2, 2)]
    y_centralised = []
    y_decentralised = []
    y_onepop = []
    y_homogeneous = []

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

        results[key][seed] = {"fitness": from_string(row["fitness"]),
                              "specialisation": from_string(row["specialisation"])}

    fig = plt.figure(figsize=(19, 9))

    cm = plt.cm.get_cmap('RdYlGn')
    norm = colours.Normalize(vmin=0, vmax=1)
    m = plt.cm.ScalarMappable(norm=norm, cmap=cm)

    for row in range(1, len(setups)+1):
        for col in range(1, max_agents//2 + 1):
            plot = ((row - 1) * max_agents//2) + col

            ax = fig.add_subplot(len(setups), max_agents//2, plot)#, sharey=True)
            setup = setups[row-1]
            num_agents = col*2
            key = f"{setup}-{num_agents}"
            plot_name = f"{key}"

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
            #ax.set_ylabel("Fitness per Agent")
            ax.set_title(plot_name)

    #fig.colorbar(m)
    plt.tight_layout(pad=3.0)
    plt.suptitle("Performance Across Episodes")
    plt.savefig(path_to_graph)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot episodes')
    parser.add_argument('--results_path', action="store")
    parser.add_argument('--graph_path', action="store")
    parser.add_argument('--max_agents', action="store")
    parser.add_argument('--y_height', action="store")

    results_path = parser.parse_args().results_path
    graph_path = parser.parse_args().graph_path
    max_agents = int(parser.parse_args().max_agents)
    y_height = parser.parse_args().y_height

    if y_height:
        y_height = int(y_height)

    plot_episodes(results_path, graph_path, max_agents, y_height)