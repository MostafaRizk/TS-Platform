import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse

from scipy import stats
from operator import add
from evaluation.plot_scalability import from_string

setups = ["Centralised", "Decentralised", "Fully-centralised"]
setup_labels = ["CTDE", "Fully Decentralised", "Fully Centralised"]
spec_metric_index = 2  # R_spec
action_names = ["F", "B", "L", "R", "P", "D"]


def load_results(path_to_results, max_agents):
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
                              "specialisation": from_string(row["specialisation"]),
                              "behaviour_characterisation": from_string(row["behaviour_characterisation"], dim=3),
                              "participation": from_string(row["participation"]),
                              "trajectory": from_string(row["trajectory"], dim=3)}

    return results


def plot_trajectory(results, path_to_graph, learning_type, num_agents, run_number, episode_numbers, max_fitness):
    key = f"{learning_type}-{num_agents}"
    all_runs = [None] * len(results[key])

    for index, seed in enumerate(results[key]):
        team_fitness_list = [0] * len(results[key][seed]["fitness"][0])

        for j in range(num_agents):
            team_fitness_list = list(map(add, team_fitness_list, results[key][seed]["fitness"][j]))

        these_episodes = [None] * len(team_fitness_list)

        # For every episode, store the fitness per agent, agent fitnesses, team spec, agent bcs, agent participations and agent trajectories
        for episode_index, episode_fitness in enumerate(team_fitness_list):
            these_episodes[episode_index] = {
                "fitness_per_agent": episode_fitness / num_agents,
                "agent_fitnesses": [results[key][seed]["fitness"][agent_id][episode_index] for agent_id in range(num_agents)],
                "team_specialisation": results[key][seed]["specialisation"][episode_index][spec_metric_index],
                "agent_bcs": [results[key][seed]["behaviour_characterisation"][agent_id][episode_index] for agent_id in range(num_agents)],
                "agent_participations": results[key][seed]["participation"][episode_index],
                "agent_trajectories": [results[key][seed]["trajectory"][agent_id][episode_index] for agent_id in range(num_agents)]
            }

        all_runs[index] = (np.mean(team_fitness_list) / num_agents, these_episodes)

    all_runs = sorted(all_runs, key=lambda x: x[0])
    all_runs[run_number] = (all_runs[run_number][0], sorted(all_runs[run_number][1], key=lambda x: x["fitness_per_agent"]))

    fig = plt.figure(figsize=(8, 9))
    label_index = setups.index(learning_type)
    plt.suptitle(f"Behaviour Inspection for {setup_labels[label_index]} w/ {num_agents}-Agents: Run {run_number}")
    num_cols = len(episode_numbers)
    num_rows = 3

    for episode_index, episode in enumerate(episode_numbers):
        data = all_runs[run_number][1][episode]
        t = [i for i in range(len(data["agent_trajectories"][0]))]
        subplot_id = episode_index + 1
        ax = fig.add_subplot(num_rows, num_cols, subplot_id)

        for agent_id in range(len(data["agent_trajectories"])):
            ax.plot(t, data["agent_trajectories"][agent_id], label=f"Agent {agent_id+1}")

        plt.title(f"Episode {episode}- Trajectory")
        plt.xlabel("Time step")
        plt.ylabel("Y Location")
        plt.ylim(-1, 8)
        plt.legend()

        # Plot actions of each agent
        subplot_id = (episode_index + 1) + (len(episode_numbers))
        ax = fig.add_subplot(num_rows, num_cols, subplot_id)

        bottom = [0]*len(action_names)

        for agent_id in range(num_agents):
            ax.bar(np.arange(len(action_names)), data["agent_bcs"][agent_id], bottom=bottom, label=f"Agent {agent_id + 1}")
            bottom = [bottom[i] + data["agent_bcs"][agent_id][i] for i in range(len(action_names))]

        ax.set_xticks(np.arange(len(action_names)))
        ax.set_xticklabels(action_names)
        plt.ylim(0, len(t)*num_agents)
        plt.ylabel("Action Count")
        plt.title(f"Episode {episode}- Agent Actions")

        # Plot fitness of each agent
        subplot_id = (episode_index + 1) + 2*(len(episode_numbers))
        ax = fig.add_subplot(num_rows, num_cols, subplot_id)

        ax.bar(np.arange(num_agents), data["agent_fitnesses"], color=[f"C{agent_id}" for agent_id in range(num_agents)])

        ax.set_xticks(np.arange(num_agents))
        ax.set_xticklabels([f"Agent {agent_id+1}" for agent_id in range(num_agents)])
        plt.ylim(-2000, max_fitness)
        plt.ylabel("Fitness")
        plt.title(f"Episode {episode}- Agent Fitness")

    fig.tight_layout(pad=3.0)
    plt.savefig(path_to_graph)

results = load_results(path_to_results="/Users/mostafa/Documents/Code/PhD/TS-Platform/results/2021_08_06_a_scalability_variable_arena/data/results_reevaluated_30ep_final.csv",
             max_agents=8)

learning_type = "Centralised"
num_agents = 8
run_number = 1
episode_numbers = [29, 0]

plot_trajectory(results=results,
                path_to_graph=f"/Users/mostafa/Documents/Code/PhD/TS-Platform/results/2021_08_06_a_scalability_variable_arena/analysis/behaviour_inspection_{learning_type}_{num_agents}_{run_number}_{episode_numbers}.png",
                learning_type=learning_type,
                num_agents=num_agents,
                run_number=run_number,
                episode_numbers=episode_numbers,
                max_fitness=40000)