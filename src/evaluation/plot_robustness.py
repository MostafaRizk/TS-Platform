import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

from evaluation.plot_scalability import from_string

setups = ["Centralised", "Decentralised", "One-pop"]


def plot_robustness(data_path, file_list, path_to_graph, plot_type, max_agents, y_height=15000, showing="fitness"):
    # Create results dictionary with keys for num_agents_removed
    results = {}

    for setup in setups:
        for team_size in range(2, max_agents+2, 2):
            key = f"{setup}-{team_size}"
            results[key] = {}

            for num_agents_removed in range(len(file_list)):
                if num_agents_removed < team_size:
                    results[key][num_agents_removed] = {}

    # For each results file, populate dictionary
    for num_agents_removed, results_file in enumerate(file_list):
        path_to_results = os.path.join(data_path, results_file)
        data = pd.read_csv(path_to_results)

        for index, row in data.iterrows():
            learning_type = row["learning_type"].capitalize()
            reward_level = row["reward_level"].capitalize()
            num_agents = row["num_agents"]

            if num_agents > max_agents:
                continue

            seed = row["seed"]
            ids_removed = row["agents_removed"]

            if not ids_removed:
                ids_removed = 0

            if learning_type == "Centralised" and reward_level == "Individual":
                key = f"One-pop-{num_agents}"
            else:
                key = f"{learning_type}-{num_agents}"

            if seed not in results[key][num_agents_removed]:
                results[key][num_agents_removed][seed] = {}

            results[key][num_agents_removed][seed][ids_removed] = {"fitness": from_string(row["fitness"]),
                                                                   "specialisation": from_string(row["specialisation"])}

    # Create figure with 2 subplots per row
    print()
    # For each team size
    # For each number of agents removed (that is valid for that team size)
    # Add data to plot
    # Save
    pass


if __name__ == "__main__":
    data_path = "/Users/mostafa/Documents/Code/PhD/TS-Platform/results/2021_04_09_b_magic_plot_combined/data"
    file_list = ["robustness_0_removed.csv",
                 "robustness_1_removed.csv",
                 "robustness_2_removed.csv",
                 "robustness_3_removed.csv"]

    plot_robustness(data_path, file_list, None, None, 4)