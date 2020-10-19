import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_performance_vs_team_size(results_file_name, graph_file_name):
    # Prepare data lists
    x = [i for i in range(2, 12, 2)]
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

        results[key][seed] += row["fitness"]

    for num_agents in range(2, 12, 2):
        team_fitnesses = []
        for seed in results[f"Team-{num_agents}"]:
            team_fitnesses += [results[f"Team-{num_agents}"][seed]]
        y_team += [np.mean(team_fitnesses)]
        yerr_team += [np.std(team_fitnesses)]

        ind_fitnesses = []
        for seed in results[f"Individual-{num_agents}"]:
            ind_fitnesses += [results[f"Individual-{num_agents}"][seed]]

        if not ind_fitnesses:
            ind_fitnesses = [0]

        y_ind += [np.mean(ind_fitnesses)]
        yerr_ind += [np.std(ind_fitnesses)]

    for key in results:
        print(f"{key}: {len(results[key])}")

    # Plot
    fig1, ax1 = plt.subplots(figsize=(12, 4))
    ax1.set_title('Fitness vs Number of Agents')
    ax1.set_ylim(-10000, 300000)
    ax1.set_ylabel('Fitness')
    ax1.set_xlabel('Number of Agents')
    plt.errorbar(x, y_team, yerr_team, label="Team")
    plt.errorbar(x, y_ind, yerr_ind, label="Individual")
    plt.legend(loc='upper right')
    plt.savefig(graph_file_name)

plot_performance_vs_team_size("results_final.csv", "performance_vs_team_size.png")