import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from operator import sub
from scipy import stats


def plot_performance_vs_team_size(results_file_name, graph_file_name, type):
    max_agents = 10

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

        if type == "mean":
            y_team += [np.mean(team_fitnesses)]
            yerr_team += [np.std(team_fitnesses)]
        elif type == "max":
            y_team += [np.max(team_fitnesses)]
        elif type == "median":
            y_team += [np.median(team_fitnesses)]
        elif type == "no_variance":
            y_team += [np.mean(team_fitnesses)]
        elif type == "error":
            y_team += [np.mean(team_fitnesses)]
            yerr_team += [stats.sem(team_fitnesses)]

        ind_fitnesses = []

        for seed in results[f"Individual-{num_agents}"]:
            ind_fitnesses += [results[f"Individual-{num_agents}"][seed]]

        if type == "mean":
            y_ind += [np.mean(ind_fitnesses)]
            yerr_ind += [np.std(ind_fitnesses)]
        elif type == "max":
            y_ind += [np.max(ind_fitnesses)]
        elif type == "median":
            y_ind+= [np.median(ind_fitnesses)]
        elif type == "no_variance":
            y_ind += [np.mean(ind_fitnesses)]
        elif type == "error":
            y_ind += [np.mean(ind_fitnesses)]
            yerr_ind += [stats.sem(ind_fitnesses)]

    for key in results:
        print(f"{key}: {len(results[key])}")

    # Plot
    fig1, ax1 = plt.subplots(figsize=(12, 8))
    ax1.set_title('Scalability of Centralised vs Decentralised Setups with Number of Agents', fontsize=20)
    ax1.set_ylim(-5000, 85000)
    ax1.set_xticks([x for x in range(2, max_agents+2, 2)])
    ax1.set_ylabel('Fitness per agent', fontsize=18)
    ax1.set_xlabel('Number of Agents', fontsize=18)

    for tick in ax1.xaxis.get_major_ticks():
        tick.label.set_fontsize(16)

    for tick in ax1.yaxis.get_major_ticks():
        tick.label.set_fontsize(16)

    if type == "mean" or type == "error":
        plt.errorbar(x, y_team, yerr_team, label="Centralised")
        plt.errorbar(x, y_ind, yerr_ind, label="Decentralised")
    else:
        plt.plot(x, y_team, label="Centralised")
        plt.plot(x, y_ind, label="Decentralised")

    plt.legend(loc='upper right', fontsize=16)
    plt.savefig(graph_file_name)


def plot_multi_setup_evolution(results_folder, graph_file):
    setups = ["Team-2", "Team-4", "Team-6", "Team-8", "Team-10", "Individual-2", "Individual-4", "Individual-6", "Individual-8", "Individual-10"]
    plots_per_col = 5
    fig1, ax1 = plt.subplots(plots_per_col, 2, figsize=(15, 15), sharex=True, sharey=True)
    fig1.suptitle('Fitness Through All Evolutionary Runs', fontsize=22)

    x = [i for i in range(0, 1020, 20)]
    y = {}
    yerr = {}
    y_all = {}

    for setup in setups:
        y[setup] = []
        yerr[setup] = []
        y_all[setup] = {}

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
            seed = row["seed"]
            key = f"{reward_level}-{num_agents}"

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

    plot_row = 0
    plot_col = 0

    for setup in setups:
        ax1[plot_row][plot_col].set_ylabel(f'Best Fitness', fontsize=18)
        ax1[plot_row][plot_col].set_xlabel(f'Generation', fontsize=18)

        team_type = None
        num_agents = setup.split('-')[1]

        if setup.split('-')[0] == "Team":
            team_type = "Centralised"
        else:
            team_type = "Decentralised"

        ax1[plot_row][plot_col].set_title(f'{team_type} team with {num_agents} agents')

        # Plot mean and alpha
        '''
        ax1[plot_row][plot_col].plot(x, y[setup], 'b-')
        y_lower = [y[setup][i] - yerr[setup][i] for i in range(len(x))]
        y_upper = [y[setup][i] + yerr[setup][i] for i in range(len(x))]
        ax1[plot_row][plot_col].fill_between(x, y_lower, y_upper, alpha=0.8)
        '''

        # Plot all seeds
        desired_length = 1020 // 20

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


def plot_fitness_comparison(results_file, graph_file):
    num_agents = 4

    # Read data
    data = pd.read_csv(results_file)

    # Format data
    results = {"Centralised": {},
               "Decentralised": {}
               }

    for index, row in data.iterrows():
        reward_level = row["reward_level"].capitalize()
        seed = row["seed"]
        num_agents_in_model = row["num_agents"]

        key = None

        if num_agents_in_model != num_agents:
            continue

        if reward_level == "Team":
            key = f"Centralised"
        elif reward_level == "Individual":
            key = f"Decentralised"

        if seed in results[key]:
            continue
        else:
            results[key][seed] = 0

        fitness = row["fitness"]

        if reward_level == "Team":
            fitness /= num_agents

        results[key][seed] += fitness

    def set_axis_style(ax, labels):
        ax.get_xaxis().set_tick_params(direction='out')
        ax.xaxis.set_ticks_position('bottom')
        ax.set_xticks(np.arange(1, len(labels) + 1))
        ax.set_xticklabels(labels, fontsize=12)
        ax.set_xlim(0.25, len(labels) + 0.75)
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(12)

        # Plot data
    slope_centralised_fitnesses = []
    for seed in results["Centralised"]:
        slope_centralised_fitnesses += [results["Centralised"][seed]]

    slope_decentralised_fitnesses = []
    for seed in results["Decentralised"]:
        slope_decentralised_fitnesses += [results["Decentralised"][seed]]

    slope_data = [slope_centralised_fitnesses, slope_decentralised_fitnesses]

    fig, (ax1) = plt.subplots(nrows=1, ncols=1, figsize=(9, 6))
    fig.suptitle('Performance of Centralised vs Decentralised Solutions', fontsize=18)

    #ax1.set_title('Slope', fontsize=16)
    ax1.set_ylabel('Fitness per agent', fontsize=14)
    ax1.set_ylim(-5000, 70000)
    parts = ax1.violinplot(slope_data)
    for pc in parts['bodies']:
        pc.set_color('blue')
    for pc in ('cbars','cmins','cmaxes'):
        parts[pc].set_color('blue')

    # set style for the axes
    labels = ['Centralised', 'Decentralised']
    for ax in [ax1]:
        set_axis_style(ax, labels)

    plt.subplots_adjust(bottom=0.2, wspace=0.05)
    plt.savefig(graph_file)

#old_seed_directory = "../../results/2020_11_22_magic_plot_combined"
#new_seed_directory = "../../results/2020_11_24_magic_plot_combined_new_seed"
#slope_directory = "../../results/2020_11_06_2-agents_slope_comparison"

'''
plot_performance_vs_team_size(f"{old_seed_directory}/results/results_final.csv", f"{old_seed_directory}/analysis/magic_plot_mean.png", type="mean")
plot_performance_vs_team_size(f"{old_seed_directory}/results/results_final.csv", f"{old_seed_directory}/analysis/magic_plot_max.png", type="max")
plot_performance_vs_team_size(f"{old_seed_directory}/results/results_final.csv", f"{old_seed_directory}/analysis/magic_plot_median.png", type="median")
plot_performance_vs_team_size(f"{old_seed_directory}/results/results_final.csv", f"{old_seed_directory}/analysis/magic_plot_no_variance.png", type="no_variance")

plot_performance_vs_team_size(f"{new_seed_directory}/results/results_final.csv", f"{new_seed_directory}/analysis/magic_plot_mean.png", type="mean")
plot_performance_vs_team_size(f"{new_seed_directory}/results/results_final.csv", f"{new_seed_directory}/analysis/magic_plot_max.png", type="max")
plot_performance_vs_team_size(f"{new_seed_directory}/results/results_final.csv", f"{new_seed_directory}/analysis/magic_plot_median.png", type="median")
plot_performance_vs_team_size(f"{new_seed_directory}/results/results_final.csv", f"{new_seed_directory}/analysis/magic_plot_no_variance.png", type="no_variance")

'''

#plot_performance_vs_team_size(f"{old_seed_directory}/results/results_final.csv", f"{old_seed_directory}/analysis/magic_plot_error.png", type="error")
#plot_performance_vs_team_size(f"{new_seed_directory}/results/results_final.csv", f"{new_seed_directory}/analysis/magic_plot_error.png", type="error")

#plot_multi_setup_evolution("../../results/2020_11_24_magic_plot_combined_new_seed/results", "../../results/2020_11_24_magic_plot_combined_new_seed/analysis/magic_plot_history_all.png")


#plot_fitness_comparison(f"{slope_directory}/results/results_final.csv", f"{slope_directory}/analysis/decentralised_comparison.png")


unique_seeding_directory = "../../results/2020_12_09_unique_seeds"
seed_1_directory = "../../results/2020_11_22_magic_plot_combined"
seed_2_directory = "../../results/2020_11_24_magic_plot_combined_new_seed"

plot_fitness_comparison(f"{unique_seeding_directory}/results/results_final.csv", f"{unique_seeding_directory}/analysis/unique_seeding.png")
plot_fitness_comparison(f"{seed_1_directory}/results/results_final.csv", f"{unique_seeding_directory}/analysis/first_seeding.png")
plot_fitness_comparison(f"{seed_2_directory}/results/results_final.csv", f"{unique_seeding_directory}/analysis/second_seeding.png")
