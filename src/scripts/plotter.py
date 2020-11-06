import copy
import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import copy

from pylab import legend
from glob import glob


def plot_hardcoded_fitness(results_file, graph_file):
    data = pd.read_csv(results_file)
    strategies = ["Generalist-Generalist", "Generalist-Dropper", "Generalist-Collector", "Dropper-Dropper", "Dropper-Collector", "Collector-Collector"]
    num_seeds = 50
    missing_strategies = [strategies[:] for x in range(num_seeds)]

    results = {"Generalist-Generalist": {"Individual 1": [], "Individual 2": []},
               "Generalist-Dropper": {"Individual 1": [], "Individual 2": []},
               "Generalist-Collector": {"Individual 1": [], "Individual 2": []},
               "Dropper-Dropper": {"Individual 1": [], "Individual 2": []},
               "Dropper-Collector": {"Individual 1": [], "Individual 2": []},
               "Collector-Collector": {"Individual 1": [], "Individual 2": []}}

    data_points = 0

    for index, row in data.iterrows():
        results[str(row["Strategy"])]["Individual 1"] += [row["Individual 1 Score"]]
        results[str(row["Strategy"])]["Individual 2"] += [row["Individual 2 Score"]]
        data_points += 1
        print(index)
        missing_strategies[int(row["Seed"])-1].remove(row["Strategy"])

    print(f"Quality check: There were {data_points} data points out of {num_seeds*len(strategies)}. ")

    for i in range(len(missing_strategies)):
        if missing_strategies[i]:
            seed = i+1
            print(f"Seed {seed}: {missing_strategies[i]}")

    # Plot data
    fig1, ax1 = plt.subplots(figsize=(12,4))
    ax1.set_title('Fitness Scores of Hardcoded Strategies')
    ax1.set_ylim(-1000, 100000)
    ax1.set_ylabel('Fitness')
    ax1.set_xlabel('Team Strategy')

    positions = [1, 4, 7, 10, 13, 16]

    for i in range(len(strategies)):
        strategy = strategies[i]
        box1 = ax1.boxplot(results[strategy]["Individual 1"], positions=[positions[i]], widths=0.3)
        box2 = ax1.boxplot(results[strategy]["Individual 2"], positions=[positions[i]+1], widths=0.3)
        for item in ['boxes', 'whiskers', 'fliers', 'medians', 'caps']:
            plt.setp(box1[item], color="red")
            plt.setp(box2[item], color="blue")

    ax1.set_xticklabels(strategies)
    ax1.set_xticks([x + 0.5 for x in positions])

    hB, = ax1.plot([1, 1], 'r-')
    hR, = ax1.plot([1, 1], 'b-')
    legend((hB, hR), ('Individual 1', 'Individual 2'))
    hB.set_visible(False)
    hR.set_visible(False)

    plt.savefig(graph_file)
    #plt.show()

def plot_evolution_fitness(results_file, graph_file):
    # Read data
    data = pd.read_csv(results_file)

    # Format data
    results = {"Speed-0": [],
               "Speed-4": []
               }

    for index, row in data.iterrows():
        #reward_level = row["reward_level"].capitalize()
        #key = f"{reward_level}"
        sliding_speed = row["sliding_speed"]
        key = f"Speed-{sliding_speed}"

        results[key] += [row["fitness"]]

    # Plot data
    fig1, ax1 = plt.subplots(figsize=(12, 4))
    ax1.set_title('Best Fitness Score of Evolved Runs')
    ax1.set_ylim(0, 130000)
    ax1.set_ylabel('Fitness')
    ax1.set_xlabel('Sliding Speed')

    positions = [0.5, 2.5]  # [1,3,5,7]
    #configs = ["Homogeneous-Individual", "Homogeneous-Team", "Heterogeneous-Individual", "Heterogeneous-Team"]
    configs = ["Speed-0", "Speed-4"]

    for i in range(len(configs)):
        config = configs[i]
        box1 = ax1.boxplot(results[config], positions=[positions[i]], widths=0.3)
        #box2 = ax1.boxplot(results[config]["Individual 2"], positions=[positions[i]+1], widths=0.3)
        for item in ['boxes', 'whiskers', 'fliers', 'medians', 'caps']:
            plt.setp(box1[item], color="red")
            #plt.setp(box2[item], color="blue")

    ax1.set_xticklabels([x for x in configs])
    #ax1.set_xticks([x + 0.5 for x in positions])
    #ax1.set_xticks([x for x in positions])

    hB, = ax1.plot([1, 1], 'r-')
    #hR, = ax1.plot([1, 1], 'b-')
    #legend((hB, hR), ('Individual 1', 'Individual 2'))
    hB.set_visible(False)
    #hR.set_visible(False)

    #plt.show()
    plt.savefig(graph_file)


def plot_evolution_specialisation(results_file, graph_file):
    # Read data
    data = pd.read_csv(results_file)
    num_seeds = 30
    #trimmed_data = data[[" Team Type", " Slope Angle", " Arena Length", " Sigma", " Population", " Fitness"]]

    # Format data
    results = {  # "Homogeneous-Individual": [],
        "Homogeneous-Team": [],
        "Heterogeneous-Individual": [],
        # "Heterogeneous-Team": []]
    }

    data_points = 0

    for index, row in data.iterrows():
        team_type = row[" Team Type"].capitalize()
        selection_level = row[" Selection Level"].capitalize()
        key = f"{team_type}-{selection_level}"

        results[key] += [row[" Specialisation"]]
        data_points += 1

    print(f"Quality check: There were {data_points} data points out of {num_seeds*len(results)}. ")

    # Plot data
    fig1, ax1 = plt.subplots(figsize=(12,4))
    ax1.set_title('Best Fitness Score of Evolved Runs')
    ax1.set_ylim(0, 1)
    ax1.set_ylabel('Specialisation')
    ax1.set_xlabel('Evolutionary Configuration')

    positions = [1, 3]
    #configs = ["Homogeneous-Individual", "Homogeneous-Team", "Heterogeneous-Individual", "Heterogeneous-Team"]
    configs = ["Homogeneous-Team", "Heterogeneous-Individual"]

    for i in range(len(configs)):
        config = configs[i]
        box1 = ax1.boxplot(results[config], positions=[positions[i]], widths=0.3)
        #box2 = ax1.boxplot(results[config]["Individual 2"], positions=[positions[i]+1], widths=0.3)
        for item in ['boxes', 'whiskers', 'fliers', 'medians', 'caps']:
            plt.setp(box1[item], color="red")
            #plt.setp(box2[item], color="blue")

    ax1.set_xticklabels([x for x in configs])
    #ax1.set_xticks([x + 0.5 for x in positions])
    ax1.set_xticks([x for x in positions])

    hB, = ax1.plot([1, 1], 'r-')
    #hR, = ax1.plot([1, 1], 'b-')
    #legend((hB, hR), ('Individual 1', 'Individual 2'))
    hB.set_visible(False)
    #hR.set_visible(False)

    #plt.show()
    plt.savefig(graph_file)


def plot_evolution_history(results_folder, graph_file):
    # Get list of fitnesses from each file
    x = [i for i in range(20,1020,20)]
    y = []
    yerr = []

    for i in range(20, 1020, 20):
        results_file = f"{results_folder}/results_{i}.csv"
        data = pd.read_csv(results_file)

        fitnesses = []
        for index, row in data.iterrows():
            fitnesses += [row["fitness"]]

        y += [np.mean(fitnesses)]
        yerr += [np.std(fitnesses)]

    # Plot
    fig1, ax1 = plt.subplots(figsize=(12, 4))
    ax1.set_title('Fitness Throughout Evolution')
    ax1.set_ylim(0, 300000)
    ax1.set_ylabel('Fitness')
    ax1.set_xlabel('Generation')
    plt.errorbar(x, y, yerr)
    plt.savefig(graph_file)


def count_results(results_file):
    # Read data
    data = pd.read_csv(results_file)

    # Format data
    results = {"Team-2": 0,
               "Team-4": 0,
               "Team-6": 0,
               "Team-8": 0,
               "Team-10": 0,
               "Individual-2": 0,
               "Individual-4": 0,
               "Individual-6": 0,
               "Individual-8": 0,
               "Individual-10": 0,
               "Team-2-slope-0": 0,
               "Individual-2-slope-0": 0
               }

    for index, row in data.iterrows():
        reward_level = row["reward_level"].capitalize()
        num_agents = row["num_agents"]
        sliding_speed = row["sliding_speed"]

        if sliding_speed == 4:
            key = f"{reward_level}-{num_agents}"
        else:
            key = f"{reward_level}-{num_agents}-slope-{sliding_speed}"

        results[key] += 1

    for key in results:
        print(f"{key}: {results[key]}")


def get_seed_dict(results_file):
    """
    Get a dict of lists of the unique seeds in the results file

    @param results_file:
    @return:
    """
    # Read data
    data = pd.read_csv(results_file)

    # Format data
    results = {"Team-2-slope-0": [],
               "Individual-2-slope-0": []
               }

    for index, row in data.iterrows():
        reward_level = row["reward_level"].capitalize()
        num_agents = row["num_agents"]
        sliding_speed = row["sliding_speed"]
        seed = row["seed"]

        if sliding_speed == 4:
            continue
        else:
            key = f"{reward_level}-{num_agents}-slope-{sliding_speed}"

        results[key] += [seed]

    return results


def find_seed_overlap(seed_dict_1, seed_dict_2):
    """
    Check if, between two sets of experiments with missing seeds, there are enough unique seeds to create a plot

    @param seed_dict_1:
    @param seed_dict_2:
    @return:
    """
    seed_dict_3 = copy.deepcopy(seed_dict_1)

    for key in seed_dict_2:
        for seed in seed_dict_2[key]:
            if seed not in seed_dict_3[key]:
                seed_dict_3[key] += [seed]

    return seed_dict_3


def get_seeds_to_rerun(original_experiment_directory, combined_seed_dict):
    # Get list of parameter files from directory
    os.chdir(original_experiment_directory)
    parameter_file_list = glob(f'cma_heterogeneous_individual*.json')

    combined_seed_list = combined_seed_dict["Individual-2-slope-0"]

    rerun_list = []

    # Iterate parameter files
    for filestring in parameter_file_list:
        # Get reward_level, num_agents, sliding_speed and seed
        filename = filestring.split("_")
        reward_level = filename[2]
        num_agents = filename[6]
        sliding_speed = filename[9]
        seed = filename[5]

        # Add parameter file to rerun list if its seed is Ind-2 and not in combined list
        if reward_level == "individual" and num_agents == "2" and sliding_speed == "0" and seed not in combined_seed_list:
            rerun_list += [filestring]


    # Print rerun list
    for i in rerun_list:
        print(i)
    print(len(rerun_list))
    # Copy parameter files to new folder

#plot_evolution_fitness("results_final.csv", "team_vs_ind.png")
#count_results("results_final.csv")

#plot_evolution_history('data/results', 'evolution_history.png')
#count_results("data/many_agents/results/results_final.csv")
#count_results("../../results/2020_10_02_CMA many agents and 0 slope/results/results_final.csv")
#count_results("../../results/2020_10_07_CMA many agents and big populations/results/results_final.csv")

#seed_dict_1 = get_seed_dict("../../results/2020_10_02_CMA many agents and 0 slope/results/results_final.csv")
#seed_dict_2 = get_seed_dict("../../results/2020_10_07_CMA many agents and big populations/results/results_final.csv")
#combined_seed_dict = find_seed_overlap(seed_dict_1, seed_dict_2)

#get_seeds_to_rerun("../../results/2020_10_02_CMA many agents and 0 slope/experiments", combined_seed_dict)

plot_evolution_fitness("../../results/2020_11_06_2-agents_slope_comparison/results/results_final.csv", "slope_vs_no_slope.png")