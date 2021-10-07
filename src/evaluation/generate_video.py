import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

from scipy import stats
from operator import add
from evaluation.plot_trajectory import load_results, plot_trajectory
from evaluation.evaluate_model import evaluate_model
from array2gif import write_gif
from PIL import Image
import moviepy.editor as mp

spec_metric_index = 0 # R_coop
#spec_metric_index = 2  # R_spec


def generate_video(results, path_to_video_folder, learning_type, num_agents, run_number, episode_numbers, agents_to_remove=0):
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
                "agent_trajectories": [results[key][seed]["trajectory"][agent_id][episode_index] for agent_id in range(num_agents)],
                "episode_order_id": episode_index
            }

        all_runs[index] = (np.mean(team_fitness_list) / num_agents, results[key][seed]["model_name"], these_episodes)

    all_runs = sorted(all_runs, key=lambda x: x[0])
    all_runs[run_number] = (all_runs[run_number][0], all_runs[run_number][1], sorted(all_runs[run_number][2], key=lambda x: x["fitness_per_agent"]))

    model_path = all_runs[run_number][1]
    total_episodes = len(all_runs[run_number][2])

    if agents_to_remove <= 0:
        evaluation_results = evaluate_model(model_path, rendering="True", episodes=total_episodes, save_video=True)

    else:
        evaluation_results = evaluate_model(model_path, rendering="True", episodes=total_episodes, ids_to_remove=[k for k in range(agents_to_remove)], save_video=True)

    video_data = evaluation_results['video_frames']

    for episode in episode_numbers:
        episode_order_id = all_runs[run_number][2][episode]["episode_order_id"]
        filename = f"video_{learning_type}_{num_agents}_{run_number}_{episode}_{agents_to_remove}.gif"
        save_to = os.path.join(path_to_video_folder, filename)
        write_gif(video_data[episode_order_id], save_to, fps=5)

        clip = mp.VideoFileClip(save_to)
        clip.write_videofile(save_to.replace(".gif", ".mp4"))


if __name__ == "__main__":
    #results = load_results(path_to_results="/Users/mostafa/Documents/Code/PhD/TS-Platform/results/2021_08_06_a_scalability_variable_arena/data/results_reevaluated_30ep_final.csv", max_agents=8)
    parent_path = "/Users/mostafa/Documents/Code/PhD/TS-Platform/results/2021_10_06_b_aamas_data/2021_10_05_scalability_variable_arena_correct_neurons"
    results = load_results(path_to_results=f"{parent_path}/data/results_reevaluated_30ep_final.csv", max_agents=8)


    configs = [{
                "learning_type": "Fully-centralised",
                "num_agents": 2,
                "run_number": 15,
                "episode_numbers": [29, 0],
                "agents_to_remove": 0},
        ]
    '''{
    "learning_type": "Decentralised",
    "num_agents": 2,
    "run_number": 15,
    "episode_numbers": [29, 0],
    "agents_to_remove": 0},
    {
    "learning_type": "Fully-centralised",
    "num_agents": 8,
    "run_number": 29,
    "episode_numbers": [29, 0],
    "agents_to_remove": 0}
    ]'''

    for config in configs:
        learning_type = config["learning_type"]
        num_agents = config["num_agents"]
        run_number = config["run_number"]
        episode_numbers = config["episode_numbers"]
        agents_to_remove = config["agents_to_remove"]

        plot_trajectory(results=results,
                        path_to_graph=f"{parent_path}/analysis/behaviour_inspection_{learning_type}_{num_agents}_{run_number}_{episode_numbers}.pdf",
                        learning_type=learning_type,
                        num_agents=num_agents,
                        run_number=run_number,
                        episode_numbers=episode_numbers,
                        max_fitness=30000)

        '''generate_video(results=results,
                        path_to_video_folder=f"{parent_path}/analysis",
                        learning_type=learning_type,
                        num_agents=num_agents,
                        run_number=run_number,
                        episode_numbers=episode_numbers,
                        agents_to_remove=agents_to_remove)'''