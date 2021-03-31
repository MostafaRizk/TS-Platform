import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from operator import sub
from scipy import stats


def plot_multi_setup_evolution(data_path, graph_file, num_generations=1000, step_size=50):
    setups = ["Centralised-2", "Centralised-4", "Centralised-6"]
    plots_per_col = 3
    fig1, ax1 = plt.subplots(plots_per_col, 2, figsize=(15, 15), sharex=True, sharey=True)
    fig1.suptitle('Fitness Through All Evolutionary Runs', fontsize=22)

    x = [i for i in range(0, num_generations+step_size, step_size)]
    y = {}
    yerr = {}
    y_all = {}

    for setup in setups:
        y[setup] = []
        yerr[setup] = []
        y_all[setup] = {}

    for i in range(0, num_generations+step_size, step_size):

        results_file = f"{data_path}/results_{i}.csv"

        if i == 0:
            results_file = f"{data_path}/results_final.csv"

        data = pd.read_csv(results_file)

        fitnesses = {}

        for setup in setups:
            fitnesses[setup] = []

        for index, row in data.iterrows():
            learning_type = row["learning_type"].capitalize()
            reward_level = row["reward_level"].capitalize()
            num_agents = row["num_agents"]
            seed = row["seed"]

            if learning_type == "Centralised" and reward_level == "Individual":
                key = f"One-pop-{num_agents}"
            else:
                key = f"{learning_type}-{num_agents}"

            if key not in setups:
                continue

            fitness = None

            if i == 0:
                fitness = row["seed_fitness"]
            else:
                fitness = row["fitness"]

            if reward_level == "Team":
                fitness /= num_agents

            fitnesses[key] += [fitness]

            if seed in y_all[key]:
                y_all[key][seed] += [fitness]
            else:
                y_all[key][seed] = [fitness]

        for setup in setups:
            y[setup] += [np.mean(fitnesses[setup])]
            yerr[setup] += [np.std(fitnesses[setup])]
            yerr[setup] += [stats.sem(fitnesses[setup])]

    plot_row = 0
    plot_col = 0

    for setup in setups:
        ax1[plot_row][plot_col].set_ylabel(f'Best Fitness', fontsize=18)
        ax1[plot_row][plot_col].set_xlabel(f'Generation', fontsize=18)

        team_type = "-".join(setup.split("-")[0:-1])
        num_agents = setup.split('-')[-1]

        ax1[plot_row][plot_col].set_title(f'{team_type} with {num_agents} agents')

        # Plot mean and alpha
        '''
        ax1[plot_row][plot_col].plot(x, y[setup], 'b-')
        y_lower = [y[setup][i] - yerr[setup][i] for i in range(len(x))]
        y_upper = [y[setup][i] + yerr[setup][i] for i in range(len(x))]
        ax1[plot_row][plot_col].fill_between(x, y_lower, y_upper, alpha=0.8)
        '''

        # Plot all seeds
        desired_length = num_generations // step_size

        for seed in y_all[setup]:
            actual_length = len(y_all[setup][seed])

            if actual_length < desired_length:
                # y_all[setup][seed] += [y_all[setup][seed][-1]] # Fill in plot with last value
                y_all[setup][seed] += [float("NaN") for i in range(desired_length - actual_length)]

        for seed in y_all[setup]:
            ax1[plot_row][plot_col].plot(x, y_all[setup][seed])

        plot_row += 1

        if plot_row >= plots_per_col:
            plot_row = 0
            plot_col += 1

    for ax in ax1.flat:
        ax.label_outer()

    plt.savefig(graph_file)

plot_multi_setup_evolution("/Users/mostafa/Documents/Code/PhD/TS-Platform/results/2021_03_22_equal_games_per_learner_sequential_reproducible/data", "generations.png")