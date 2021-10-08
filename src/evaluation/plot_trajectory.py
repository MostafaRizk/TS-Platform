import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse

from scipy import stats
from operator import add
from evaluation.plot_scalability import from_string

setups = ["Centralised", "Decentralised", "Fully-centralised"]
setup_labels = ["CTDE", "Fully Decentralised", "Fully Centralised"]
spec_metric_index = 0 # R_coop
#spec_metric_index = 2  # R_spec
action_names = ["F", "B", "L", "R", "P", "D"]

# Plot
suptitle_font = 20
title_font = 17.5
label_font = 20
tick_font = 15
legend_font = 15
label_padding = 1.08


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
                              "total_resources": from_string(row["total_resources"], dim=1),
                              "trajectory": from_string(row["trajectory"], dim=3),
                              "model_name": row["model_name"]}

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
                "total_resources": results[key][seed]["total_resources"][episode_index],
                "agent_trajectories": [results[key][seed]["trajectory"][agent_id][episode_index] for agent_id in range(num_agents)]
            }

        all_runs[index] = (np.mean(team_fitness_list) / num_agents, these_episodes)

    all_runs = sorted(all_runs, key=lambda x: x[0])
    all_runs[run_number] = (all_runs[run_number][0], sorted(all_runs[run_number][1], key=lambda x: x["fitness_per_agent"]))

    fig = plt.figure(figsize=(8, 9))
    label_index = setups.index(learning_type)
    plt.suptitle(f"{setup_labels[label_index]} Behaviour Inspection:\n"
                 f"{num_agents} Agents - Run {run_number}", fontsize=suptitle_font)
    num_cols = len(episode_numbers)
    #num_rows = 4
    num_rows = 2
    linewidth=3

    for episode_index, episode in enumerate(episode_numbers):
        data = all_runs[run_number][1][episode]

        # Plot trajectory of each agent
        t = [i for i in range(len(data["agent_trajectories"][0]))]
        subplot_id = episode_index + 1
        ax = fig.add_subplot(num_rows, num_cols, subplot_id)

        for agent_id in range(len(data["agent_trajectories"])):
            ax.plot(t, data["agent_trajectories"][agent_id], label=f"Agent {agent_id+1}", linewidth=linewidth)

        plt.title(f"Episode {episode}- Trajectory", fontsize=title_font, y=label_padding)
        plt.xlabel("Time step", fontsize=label_font)
        if episode_index == 0:
            plt.ylabel("Y Location", fontsize=label_font)

        if subplot_id % num_cols != 1:
            ax.set_yticklabels([])

        plt.ylim(-1, 8)
        plt.setp(ax.get_xticklabels(), fontsize=tick_font)
        plt.setp(ax.get_yticklabels(), fontsize=tick_font)
        #plt.legend(fontsize=legend_font)

        '''
        # Plot actions of each agent
        subplot_id = (episode_index + 1) + (len(episode_numbers))
        ax = fig.add_subplot(num_rows, num_cols, subplot_id)
        #bottom = [0]*len(action_names)
        x = np.arange(len(action_names))
        width = 0.35/(num_agents/2)

        for agent_id in range(num_agents):
            action_list = []
            for action_index in range(len(action_names)):
                action_list += [data["agent_bcs"][agent_id][action_index]]

            position = x - (width*num_agents/2) + (2*agent_id + 1)*width/2
            ax.bar(position, action_list, width=width, label=f"Agent {agent_id + 1}")
            #bottom = [bottom[i] + data["agent_bcs"][agent_id][i] for i in range(len(action_names))]

        ax.set_xticks(np.arange(len(action_names)))
        ax.set_xticklabels(action_names, fontsize=tick_font)
        #plt.ylim(0, len(t)*num_agents)
        plt.ylim(0, len(t) * 0.75)
        if episode_index == 0:
            plt.ylabel("Action Count", fontsize=label_font)
        plt.title(f"Episode {episode}- Agent Actions", fontsize=title_font, y=label_padding)'''

        # Plot fitness of each agent

        # Use this when plotting only two
        subplot_id = (episode_index + 1) + (len(episode_numbers))

        # Use this when plotting all subplots
        #subplot_id = (episode_index + 1) + 2*(len(episode_numbers))

        ax = fig.add_subplot(num_rows, num_cols, subplot_id)
        ax.bar(np.arange(num_agents), data["agent_fitnesses"], color=[f"C{agent_id}" for agent_id in range(num_agents)])
        ax.set_xticks(np.arange(num_agents))
        ax.set_xticklabels([f"Agent {agent_id+1}" for agent_id in range(num_agents)], fontsize=tick_font)
        plt.ylim(-2000, max_fitness)
        if episode_index == 0:
            plt.ylabel("Fitness", fontsize=label_font)
        if subplot_id % num_cols != 1:
            ax.set_yticklabels([])
        plt.setp(ax.get_yticklabels(), fontsize=tick_font)
        plt.title(f"Episode {episode}- Agent Fitness", fontsize=title_font, y=label_padding)


        '''# Plot participation of each agent
        subplot_id = (episode_index + 1) + 3*len(episode_numbers)
        ax = fig.add_subplot(num_rows, num_cols, subplot_id)
        total = data["total_resources"]
        if total != 0:
            participation_percentage = [p/total for p in data["agent_participations"]]
        else:
            participation_percentage = data["agent_participations"]
        ax.bar(np.arange(num_agents), participation_percentage, color=[f"C{agent_id}" for agent_id in range(num_agents)])
        ax.set_xticks(np.arange(num_agents))
        ax.set_xticklabels([f"Agent {agent_id + 1}" for agent_id in range(num_agents)], fontsize=tick_font)
        plt.ylim(0, 1)
        if episode_index == 0:
            plt.ylabel("Participation", fontsize=label_font)
        plt.title(f"Episode {episode}- Agent Participation", fontsize=title_font, y=label_padding)'''

    plt.tight_layout(pad=1.0)
    plt.savefig(path_to_graph)


if __name__ == "__main__":
    results = load_results(path_to_results="/Users/mostafa/Documents/Code/PhD/TS-Platform/results/2021_08_06_a_scalability_variable_arena/data/results_reevaluated_30ep_final.csv", max_agents=8)

    learning_type = "Decentralised"
    num_agents = 8
    run_number = 29
    episode_numbers = [29, 0]

    plot_trajectory(results=results,
                    path_to_graph=f"/Users/mostafa/Documents/Code/PhD/TS-Platform/results/2021_08_06_a_scalability_variable_arena/analysis/behaviour_inspection_{learning_type}_{num_agents}_{run_number}_{episode_numbers}.png",
                    learning_type=learning_type,
                    num_agents=num_agents,
                    run_number=run_number,
                    episode_numbers=episode_numbers,
                    max_fitness=40000)