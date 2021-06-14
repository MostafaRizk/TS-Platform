import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from operator import sub
from scipy import stats
from glob import glob


def generate_results_from_numpy(path_to_data_folder):
    model_files = glob(f'{path_to_data_folder}/*.npy')

    # Create final results file
    results_file = os.path.join(path_to_data_folder, f'results.csv')
    f = open(results_file, 'w')

    header = "learning_type,algorithm_selected,team_type,reward_level,agent_type,environment,seed,num_agents,hall_size,start_zone_size,episode_length,num_episodes,architecture,bias,hidden_layers,hidden_units_per_layer,activation_function,agent_population_size,distribution,mean,variance,fitness,model_name"
    f.write(header)
    f.write("\n")

    # For every model, extract parameters and convert to a comma separated list
    for model_name in model_files:
        learning_type = model_name.split("/")[-1].split("_")[0]
        if learning_type == "centralised":
            parameter_list = model_name.split("/")[-1].split("_")[:-1]
            parameter_filename = "_".join(parameter_list) + ".json"
        else:
            raise RuntimeError("Model prefix must be Centralised or Decentralised")

        fitness = model_name.split("_")[-1].strip(".npy")

        # Log parameters and score to results file
        parameters_to_log = parameter_list + [fitness] + [model_name]
        line_to_log = ",".join(parameters_to_log)
        f.write(line_to_log)
        f.write("\n")

        # Close results file
    f.close()


def plot_fitness_comparison(rwg_results, cma_results, graph_file):
    num_agents = 2

    # Read data
    data = pd.concat([pd.read_csv(rwg_results), pd.read_csv(cma_results)])

    # Format data
    results = {"rwg": {},
               "cma": {}
               }

    for index, row in data.iterrows():
        reward_level = row["reward_level"].capitalize()
        seed = row["seed"]
        num_agents_in_model = row["num_agents"]
        algorithm = row["algorithm_selected"]
        key = algorithm

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
    rwg_fitnesses = []
    for seed in results["rwg"]:
        rwg_fitnesses += [results["rwg"][seed]]

    cma_fitnesses = []
    for seed in results["cma"]:
        cma_fitnesses += [results["cma"][seed]]

    results_data = [rwg_fitnesses, cma_fitnesses]

    fig, (ax1) = plt.subplots(nrows=1, ncols=1, figsize=(9, 6))
    fig.suptitle('Performance of RWG vs CMA', fontsize=18)

    #ax1.set_title('Slope', fontsize=16)
    ax1.set_ylabel('Fitness per agent', fontsize=14)
    ax1.set_ylim(0, 20)
    parts = ax1.violinplot(results_data)
    for pc in parts['bodies']:
        pc.set_color('blue')
    for pc in ('cbars','cmins','cmaxes'):
        parts[pc].set_color('blue')

    # set style for the axes
    labels = ['RWG', 'CMA']
    for ax in [ax1]:
        set_axis_style(ax, labels)

    plt.subplots_adjust(bottom=0.2, wspace=0.05)
    plt.savefig(graph_file)

rwg_results = "/Users/mostafa/Documents/Code/PhD/TS-Platform/results/2021_06_09_b_rwg_30_runs/data/results.csv"
cma_results = "/Users/mostafa/Documents/Code/PhD/TS-Platform/results/2021_06_09_c_cma_30_runs/data/results_final.csv"
#generate_results_from_numpy("/Users/mostafa/Documents/Code/PhD/TS-Platform/results/2021_06_09_b_rwg_30_runs/data")
plot_fitness_comparison(rwg_results, cma_results, "cma_vs_rwg.png")