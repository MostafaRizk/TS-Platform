import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse

from scipy import stats

setups = ["Centralised", "Decentralised", "One-pop"]

def plot_scalability(path_to_results, path_to_graph, plot_type, max_agents, y_height=15000, showing="fitness"):
    # Prepare data lists
    x = [i for i in range(2, max_agents+2, 2)]
    y_centralised = []
    y_decentralised = []
    y_onepop = []

    yerr_centralised = []
    yerr_decentralised = []
    yerr_onepop = []

    # Read data from results file
    data = pd.read_csv(path_to_results)

    results = {"Centralised-2": {},
               "Centralised-4": {},
               "Centralised-6": {},
               "Decentralised-2": {},
               "Decentralised-4": {},
               "Decentralised-6": {},
               "One-pop-2": {},
               "One-pop-4": {},
               "One-pop-6": {}
               }

    for index, row in data.iterrows():
        learning_type = row["learning_type"].capitalize()
        reward_level = row["reward_level"].capitalize()
        num_agents = row["num_agents"]
        seed = row["seed"]

        if learning_type == "Centralised" and reward_level == "Individual":
            key = f"One-pop-{num_agents}"
        else:
            key = f"{learning_type}-{num_agents}"

        if seed not in results[key]:
            results[key][seed] = 0

        # At the end of the loop, contains team fitness for Centralised and Decentralised
        # but only individual fitness for One-pop
        if showing == "fitness":
            results[key][seed] += row["fitness"]
        elif showing == "specialisation":
            results[key][seed] = row["specialisation"]

    scores = {"Centralised-2": [],
               "Centralised-4": [],
               "Centralised-6": [],
               "Decentralised-2": [],
               "Decentralised-4": [],
               "Decentralised-6": [],
               "One-pop-2": [],
               "One-pop-4": [],
               "One-pop-6": []
               }

    for num_agents in range(2, max_agents+2, 2):
        for setup in setups:
            key = f"{setup}-{num_agents}"

            for seed in results[key]:
                if showing == "fitness":
                    if setup == "Centralised" or setup == "Decentralised":
                        fitness = results[key][seed] / num_agents
                    else:
                        fitness = results[key][seed]

                    scores[key] += [fitness]

                elif showing == "specialisation":
                    specialisation = results[key][seed]
                    scores[key] += [specialisation]

            if setup == "Centralised":
                y = y_centralised
                yerr = yerr_centralised

            elif setup == "Decentralised":
                y = y_decentralised
                yerr = yerr_decentralised

            elif setup == "One-pop":
                y = y_onepop
                yerr = yerr_onepop

            if plot_type == "mean":
                y += [np.mean(scores[key])]
                yerr += [np.std(scores[key])]
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
    ''''''
    fig1, ax1 = plt.subplots(figsize=(12, 8))
    ax1.set_title(f'Scalability of Evolved {showing.capitalize()} with Number of Agents', fontsize=20)
    if showing == "fitness":
        ax1.set_ylim(0, y_height)
        ax1.set_ylabel('Fitness per agent', fontsize=18)
    elif showing == "specialisation":
        ax1.set_ylim(0,1.1)
        ax1.set_ylabel('Team Specialisation', fontsize=18)
    ax1.set_xticks([x for x in range(2, max_agents+2, 2)])
    ax1.set_xlabel('Number of Agents', fontsize=18)

    for tick in ax1.xaxis.get_major_ticks():
        tick.label.set_fontsize(16)

    for tick in ax1.yaxis.get_major_ticks():
        tick.label.set_fontsize(16)

    if plot_type == "mean" or plot_type == "error":
        plt.errorbar(x, y_centralised, yerr_centralised, fmt='r-', label="Centralised")
        plt.errorbar(x, y_decentralised, yerr_decentralised, fmt='b-', label="Decentralised")
        plt.errorbar(x, y_onepop, yerr_onepop, fmt='g-', label="One-pop")

    elif plot_type == "best" or plot_type == "median":
        plt.plot(x, y_centralised, 'ro-', label=f"Centralised ({plot_type})")
        plt.plot(x, y_decentralised, 'bo-', label=f"Decentralised ({plot_type})")
        plt.plot(x, y_onepop, 'go-', label=f"One-pop ({plot_type})")

    else:
        plt.plot(x, y_centralised, 'r-', label="Centralised")
        plt.plot(x, y_decentralised, 'b-', label="Decentralised")
        plt.plot(x, y_onepop, 'g-', abel="One-pop")

    plt.legend(loc='upper right', fontsize=16)
    plt.savefig(path_to_graph)


    '''
    def set_axis_style(ax, labels):
        ax.get_xaxis().set_tick_params(direction='out')
        ax.xaxis.set_ticks_position('bottom')
        ax.set_xticks(np.arange(1, len(labels) + 1))
        ax.set_xticklabels(labels, fontsize=12)
        ax.set_xlim(0.25, len(labels) + 0.75)
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(12)

    #fig = plt.figure(figsize=(19, 9))
    fig, axs = plt.subplots(1, 3, sharey=True, figsize=(19,9))
    num_cols = 3

    for col,ax in enumerate(axs):

        if showing == "fitness":
            ax.set_ylim(-2000, 15000)
            ax.set_ylabel("Fitness per Agent")
        elif showing == "specialisation":
            ax.set_ylim(-0.2, 1.2)
            ax.set_ylabel("Team Specialisation")

        num_agents = (col+1) * 2
        parts = ax.violinplot([scores[f"{setup}-{num_agents}"] for setup in setups])

        for id,pc in enumerate(parts['bodies']):
            if setups[id] == "Centralised":
                pc.set_color('red')
            elif setups[id] == "Decentralised":
                pc.set_color('blue')
            elif setups[id] == "One-pop":
                pc.set_color('green')

        for pc in ('cbars', 'cmins', 'cmaxes'):
            parts[pc].set_color('black')

        set_axis_style(ax, setups)
        ax.set_title(f"{num_agents} Agents")

    plt.suptitle(f"{showing.capitalize()} Distribution")
    plt.savefig(path_to_graph)
    '''


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot scalability')
    parser.add_argument('--results_path', action="store")
    parser.add_argument('--graph_path', action="store")
    parser.add_argument('--plot_type', action="store")
    parser.add_argument('--max_agents', action="store")
    parser.add_argument('--y_height', action="store")
    parser.add_argument('--showing', action="store")

    results_path = parser.parse_args().results_path
    graph_path = parser.parse_args().graph_path
    plot_type = parser.parse_args().plot_type
    max_agents = int(parser.parse_args().max_agents)
    y_height = parser.parse_args().y_height

    if y_height:
        y_height = int(y_height)

    showing = parser.parse_args().showing

    plot_scalability(results_path, graph_path, plot_type, max_agents, y_height, showing)