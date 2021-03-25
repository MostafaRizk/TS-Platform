import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse

from scipy import stats


def plot_scalability(path_to_results, graph_file_name, plot_type, max_agents, showing="fitness"):
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

    for num_agents in range(2, max_agents+2, 2):
        for setup in ["Centralised", "Decentralised", "One-pop"]:
            score = []
            key = f"{setup}-{num_agents}"

            for seed in results[key]:
                if showing == "fitness":
                    if setup == "Centralised" or setup == "Decentralised":
                        fitness = results[key][seed] / num_agents
                    else:
                        fitness = results[key][seed]

                    score += [fitness]

                elif showing == "specialisation":
                    specialisation = results[key][seed]
                    score += [specialisation]

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
                y += [np.mean(score)]
                yerr += [np.std(score)]
            elif plot_type == "max":
                y += [np.max(score)]
            elif plot_type == "median":
                y += [np.median(score)]
            elif plot_type == "no_variance":
                y += [np.mean(score)]
            elif plot_type == "error":
                y += [np.mean(score)]
                yerr += [stats.sem(score)]

    for key in results:
        print(f"{key}: {len(results[key])}")

    # Plot
    fig1, ax1 = plt.subplots(figsize=(12, 8))
    ax1.set_title(f'Scalability of Evolved {showing.capitalize()} with Number of Agents', fontsize=20)
    if showing == "fitness":
        ax1.set_ylim(0, 8000)
        ax1.set_ylabel('Fitness per agent', fontsize=18)
    elif showing == "specialisation":
        ax1.set_ylim(0,1)
        ax1.set_ylabel('Team Specialisation', fontsize=18)
    ax1.set_xticks([x for x in range(2, max_agents+2, 2)])
    ax1.set_xlabel('Number of Agents', fontsize=18)

    for tick in ax1.xaxis.get_major_ticks():
        tick.label.set_fontsize(16)

    for tick in ax1.yaxis.get_major_ticks():
        tick.label.set_fontsize(16)

    if plot_type == "mean" or plot_type == "error":
        plt.errorbar(x, y_centralised, yerr_centralised, label="Centralised")
        plt.errorbar(x, y_decentralised, yerr_decentralised, label="Decentralised")
        plt.errorbar(x, y_onepop, yerr_onepop, label="One-pop")
    else:
        plt.plot(x, y_centralised, label="Centralised")
        plt.plot(x, y_decentralised, label="Decentralised")
        plt.plot(x, y_onepop, label="One-pop")

    plt.legend(loc='upper right', fontsize=16)
    plt.savefig(graph_file_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot scalability')
    parser.add_argument('--results_path', action="store")
    parser.add_argument('--graph_path', action="store")
    parser.add_argument('--plot_type', action="store")
    parser.add_argument('--max_agents', action="store")
    parser.add_argument('--showing', action="store")

    results_path = parser.parse_args().results_path
    graph_path = parser.parse_args().graph_path
    plot_type = parser.parse_args().plot_type
    max_agents = int(parser.parse_args().max_agents)
    showing = parser.parse_args().showing

    plot_scalability(results_path, graph_path, plot_type, max_agents, showing)