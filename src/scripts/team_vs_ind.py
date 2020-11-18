import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from operator import sub


def plot_performance_vs_team_size(results_file_name, graph_file_name):
    max_agents = 6

    # Prepare data lists
    x = [i for i in range(2, max_agents+2, 2)]
    y_team = []
    y_ind = []
    yerr_team = []
    yerr_ind = []

    # Read data from results file
    data = pd.read_csv(results_file_name)

    results = {"Team-2": {},
               "Team-4": {},
               "Team-6": {},
               "Team-8": {},
               "Team-10": {},
               "Individual-2": {},
               "Individual-4": {},
               "Individual-6": {},
               "Individual-8": {},
               "Individual-10": {},
               #"Team-2-slope-0": {},
               #"Individual-2-slope-0": {}
               }

    for index, row in data.iterrows():
        reward_level = row["reward_level"].capitalize()
        num_agents = row["num_agents"]
        sliding_speed = row["sliding_speed"]
        seed = row["seed"]

        if sliding_speed == 4:
            key = f"{reward_level}-{num_agents}"
        else:
        #    key = f"{reward_level}-{num_agents}-slope-{sliding_speed}"
            continue

        if seed not in results[key]:
            results[key][seed] = 0

        fitness = row["fitness"]

        if reward_level == "Team":
            fitness /= num_agents

        results[key][seed] += fitness

    for num_agents in range(2, max_agents+2, 2):
        team_fitnesses = []
        for seed in results[f"Team-{num_agents}"]:
            team_fitnesses += [results[f"Team-{num_agents}"][seed]]
        y_team += [np.mean(team_fitnesses)]
        yerr_team += [np.std(team_fitnesses)]

        ind_fitnesses = []
        for seed in results[f"Individual-{num_agents}"]:
            ind_fitnesses += [results[f"Individual-{num_agents}"][seed]]

        #if not ind_fitnesses:
        #    ind_fitnesses = [0]

        y_ind += [np.mean(ind_fitnesses)]
        yerr_ind += [np.std(ind_fitnesses)]

    for key in results:
        print(f"{key}: {len(results[key])}")

    # Plot
    fig1, ax1 = plt.subplots(figsize=(12, 4))
    ax1.set_title('Fitness vs Number of Agents')
    ax1.set_ylim(-10000, 200000)
    ax1.set_xticks([x for x in range(2, max_agents+2, 2)])
    ax1.set_ylabel('Fitness')
    ax1.set_xlabel('Number of Agents')
    plt.errorbar(x, y_team, yerr_team, label="Centralised")
    plt.errorbar(x, y_ind, yerr_ind, label="Decentralised")

    '''
    plt.plot(x, y_team, 'r-', label="Team")
    y_team_lower = [y_team[i] - yerr_team[i] for i in range(len(x))]
    y_team_upper = [y_team[i] + yerr_team[i] for i in range(len(x))]
    plt.fill_between(x, y_team_lower, y_team_upper, 'r')

    plt.plot(x, y_ind, 'b-', label="Individual")
    y_ind_lower = [y_ind[i] - yerr_ind[i] for i in range(len(x))]
    y_ind_upper = [y_ind[i] + yerr_ind[i] for i in range(len(x))]
    plt.fill_between(x, y_ind_lower, y_ind_upper, 'b')
    '''

    plt.legend(loc='upper right')
    plt.savefig(graph_file_name)


def plot_multi_setup_evolution(results_folder, graph_file):
    setups = ["Team-2", "Team-4", "Team-6", "Individual-2", "Individual-4", "Individual-6"]
    fig1, ax1 = plt.subplots(3, 2, figsize=(12, 6), sharex=True, sharey=True)
    fig1.suptitle('Evolution History')

    x = [i for i in range(0, 1020, 20)]
    y = {}
    yerr = {}

    for setup in setups:
        y[setup] = []
        yerr[setup] = []

    for i in range(0, 1020, 20):

        results_file = f"{results_folder}/results_{i}.csv"

        if i == 0:
            results_file = f"{results_folder}/results_final.csv"

        data = pd.read_csv(results_file)

        fitnesses = {}

        for setup in setups:
            fitnesses[setup] = []

        for index, row in data.iterrows():
            reward_level = row["reward_level"].capitalize()
            num_agents = row["num_agents"]
            key = f"{reward_level}-{num_agents}"

            fitness = None

            if i == 0:
                fitness = row["seed_fitness"]
            else:
                fitness = row["fitness"]

            if reward_level == "Team":
                fitness /= num_agents

            fitnesses[key] += [fitness]



        for setup in setups:
            y[setup] += [np.mean(fitnesses[setup])]
            yerr[setup] += [np.std(fitnesses[setup])]

    plot_row = 0
    plot_col = 0

    for setup in setups:
        ax1[plot_row][plot_col].set_ylabel(f'Fitness')
        ax1[plot_row][plot_col].set_xlabel(f'Generation')
        #ax1[plot_row][plot_col].errorbar(x, y[setup], yerr[setup])

        ax1[plot_row][plot_col].plot(x, y[setup], 'b-')
        y_lower = [y[setup][i] - yerr[setup][i] for i in range(len(x))]
        y_upper = [y[setup][i] + yerr[setup][i] for i in range(len(x))]

        ax1[plot_row][plot_col].fill_between(x, y_lower, y_upper, alpha=0.8)
        ax1[plot_row][plot_col].set_title(f'{setup}')

        plot_row += 1

        if plot_row >= 3:
            plot_row = 0
            plot_col += 1

    for ax in ax1.flat:
        ax.label_outer()

    plt.savefig(graph_file)


plot_performance_vs_team_size("../../results/2020_11_10_magic_plot_shortened_episodes_evolution/results/results_final.csv", "magic_plot.png")
#plot_multi_setup_evolution("../../results/2020_11_10_magic_plot_shortened_episodes_evolution/results", "magic_plot_history.png")