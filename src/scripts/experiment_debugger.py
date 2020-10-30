import os
from glob import glob
import re
import pandas as pd
from shutil import copyfile
import matplotlib.pyplot as plt
import numpy as np
import json


from fitness import FitnessCalculator
from agents.nn_agent_lean import NNAgent
from glob import glob


def get_all_files():
    return [name.replace('data/experiments/', '') for name in glob(f'data/experiments/cma*json')]


# Find parameter files that weren't run
def get_unstarted_files():
    f = open('data/cma.out')
    output_file_data = f.read()
    return re.findall(r'cma.*json', output_file_data)


# Get number of seg faults
def count_seg_faults():
    f = open('data/cma.out')
    output_file_data = f.read()
    return re.findall(r'Segmentation fault', output_file_data)


# Get number of timeouts
def count_timeouts():
    f = open('data/cma.out')
    output_file_data = f.read()
    return re.findall(r'DUE TO TIME LIMIT', output_file_data)


# Get parameter files for completed runs
def get_completed_files():
    '''
    data = pd.read_csv('data/results/results_final.csv')
    complete_parameter_files = []

    for index, row in data.iterrows():
        parameter_file = '_'.join(row["model_name"].split('_')[0:-3]) + '.json'

        if parameter_file not in complete_parameter_files:
            complete_parameter_files += [parameter_file]

    return complete_parameter_files
    '''
    data = pd.read_csv('data/results/results_final.csv')
    logged_files = []
    completed_files = []

    for index, row in data.iterrows():
        logged_files += ['_'.join(row["model_name"].split('_')[0:-3]) + '.json']

    for parameter_file in all_parameter_files:
        if parameter_file in logged_files:
            completed_files += [parameter_file]

    return completed_files


# Find parameter files that seg-faulted, ran out of time or didn't run at all
def get_rerun_files(all_parameter_files, complete_parameter_files):
    rerun_parameter_files = []

    for parameter_file in all_parameter_files:
        if parameter_file not in complete_parameter_files:
            rerun_parameter_files += [parameter_file]

    return rerun_parameter_files


# Copy all the files that need re-running to a new folder and add them to LIST_cma
def copy_rerun_files(rerun_parameter_files):
    g = open(f"../LIST_cma", "a")

    for parameter_file in rerun_parameter_files:
        src = 'data/experiments/' + parameter_file
        dst = 'data/new_experiments/' + parameter_file
        copyfile(src, dst)
        g.write(f"python3 experiment.py --parameters {parameter_file}\n")

    g.close()


# Find files that started but didn't finish (seg-fault or ran out of time)
def get_incomplete_files(unstarted_parameter_files, rerun_parameter_files):
    incomplete_files = []

    for parameter_file in rerun_parameter_files:
        if parameter_file not in unstarted_parameter_files:
            incomplete_files += [parameter_file]

    return incomplete_files


def get_generation_count(incomplete_files):
    final_gen = {}
    y_vals = {}
    y_err_vals = {}

    for parameter_file in incomplete_files:
        final_gen[parameter_file] = -1
        y_vals[parameter_file] = []
        y_err_vals[parameter_file] = []

    for i in range(20, 1020, 20):
        results_file = f"data/results/results_{i}.csv"
        data = pd.read_csv(results_file)

        for index, row in data.iterrows():
            parameter_file = '_'.join(row["model_name"].split('_')[0:-3]) + '.json'

            if parameter_file in final_gen:
                final_gen[parameter_file] = i
                fitness = row["fitness"]
                y_vals[parameter_file] += [fitness]
                y_err_vals[parameter_file] += [0]


    return final_gen, y_vals, y_err_vals


def plot_evolution_history(results_folder, graph_file, num_generations, start_generation, step_size, y_min, y_max):
    # Get list of fitnesses from each file
    x = [i for i in range(0, num_generations+1, step_size)]
    y = []
    yerr = [0]*(num_generations+1)
    fitnesses = {}

    # Add seed fitness to list
    results_file = f"{results_folder}/results_{start_generation}.csv"
    data = pd.read_csv(results_file)
    for index, row in data.iterrows():
        if row["reward_level"] == "individual":
            genome_name = '_'.join(row["model_name"].split('_')[0:-2])
            fitnesses[genome_name] = [row["seed_fitness"]]

    # Add top n individual fitnesses to their respective lists
    for i in range(start_generation, num_generations+step_size, step_size):
        results_file = f"{results_folder}/results_{i}.csv"
        data = pd.read_csv(results_file)

        for index, row in data.iterrows():
            genome_name = '_'.join(row["model_name"].split('_')[0:-2])
            fitnesses[genome_name] += [row["fitness"]]

    # Plot
    fig1, ax1 = plt.subplots(figsize=(12, 4))
    ax1.set_title('Fitness Throughout Evolution')
    ax1.set_ylim(y_min, y_max)
    ax1.set_ylabel('Fitness')
    ax1.set_xlabel('Generation')
    for genome_name in fitnesses:
        plt.errorbar(x, fitnesses[genome_name], yerr)
    plt.savefig(graph_file)


def plot_evolution_history_average(results_folder, graph_file, num_generations, start_generation, step_size, y_min, y_max):
    # Get list of fitnesses from each file
    x = [i for i in range(0, num_generations+2, step_size)]
    y = []
    yerr = []

    # Add seed fitness to list
    results_file = f"{results_folder}/results_{start_generation}.csv"
    data = pd.read_csv(results_file)
    for index, row in data.iterrows():
        if row["reward_level"] == "individual":
            y += [row["seed_fitness"]]
            yerr += [0]
            break

    for i in range(start_generation, num_generations+step_size, step_size):
        results_file = f"{results_folder}/results_{i}.csv"
        data = pd.read_csv(results_file)

        fitnesses = []
        for index, row in data.iterrows():
            if row["reward_level"] == "individual":
                fitnesses += [row["fitness"]]

        y += [np.mean(fitnesses)]
        yerr += [np.std(fitnesses)]

    # Plot
    fig1, ax1 = plt.subplots(figsize=(12, 4))
    ax1.set_title('Fitness Throughout Evolution')
    ax1.set_ylim(y_min, y_max)
    ax1.set_ylabel('Fitness')
    ax1.set_xlabel('Generation')
    plt.errorbar(x, y, yerr)
    plt.savefig(graph_file)


def fix_results(results_folder, start_generation, num_generations, step_size):
    correct_header = "algorithm_selected,team_type,reward_level,agent_type,environment,seed,num_agents,num_resources,sensor_range,sliding_speed,arena_length,arena_width,cache_start,slope_start,source_start,base_cost,upward_cost_factor,downward_cost_factor,carry_factor,resource_reward,episode_length,num_episodes,architecture,bias,hidden_layers,hidden_units_per_layer,activation_function,agent_population_size,sigma,generations,tolx,tolfunhist,seed_fitness,fitness,model_name"

    for i in range(start_generation, num_generations+1, step_size):
        results_file = f"{results_folder}/results_{i}.csv"

        f = open(results_file)
        file_text = f.read().split('\n')
        f.close()
        file_text[0] = correct_header
        file_text = '\n'.join(file_text)
        f = open(results_file, 'w')
        f.write(file_text)
        f.close()

    results_file = f"{results_folder}/results_final.csv"

    f = open(results_file)
    file_text = f.read().split('\n')
    f.close()
    file_text[0] = correct_header
    file_text = '\n'.join(file_text)
    f = open(results_file, 'w')
    f.write(file_text)
    f.close()


def get_genomes(reward_level, num_genomes, num_episodes=20):
    '''
    Gets top genomes for each team setup e.g. top 30 for 2-agent, 4-agent, 6-agent, 8-agent and 10-agent

    @param reward_level:
    @param num_genomes:
    @return:
    '''
    genome_dictionary = {2: [], 4: [], 6: [], 8: [], 10: []}

    # Find relevant all_genomes file for each setup
    "all_genomes_rwg_heterogeneous_individual_nn_slope_1_1_4_1_4_8_4_1_3_7_1_3.0_0.2_2_1000_500_20_rnn_False_1_4_tanh_1000_normal_0_1_.csv"
    all_genomes_files = glob(f"all_genomes_rwg_heterogeneous*csv")

    for filename in all_genomes_files:
        # Read the genomes
        if filename.split("_")[4] != reward_level:
            continue

        num_agents = int(filename.split("_")[9])//2

        f = open(filename, "r")
        data = f.read().strip().split("\n")
        genome_list = []
        for row in data:
            genome = np.array([float(element) for element in row.split(",")[0:-num_episodes]])
            episode_scores = [float(score) for score in row.split(",")[-num_episodes:]]
            mean_score = np.mean(episode_scores)
            genome_list += [(genome, mean_score)]

        # Sort the genomes
        genome_list.sort(key=lambda x: x[1])

        # Retrieve the top num_genomes
        genome_dictionary[num_agents] += [tup[0] for tup in genome_list[-num_genomes:]]
    return genome_dictionary


def calculate_agent_fitness_distribution(reward_level, num_genomes, samples_per_genome, logfile_name):
    ind_parameter_files = {
        2: "cma_heterogeneous_individual_nn_slope_4005303369_2_4_1_0_8_4_1_3_7_1_1.0_1.0_2_1000_500_20_rnn_False_1_4_tanh_100_0.2_1000_0.001_200.0.json",
        4: "cma_heterogeneous_individual_nn_slope_550290314_4_8_1_4_8_8_1_3_7_1_3.0_0.2_2_1000_500_20_rnn_False_1_4_tanh_100_0.2_1000_0.001_200.0.json",
        6: "cma_heterogeneous_individual_nn_slope_1298508492_6_12_1_4_8_12_1_3_7_1_3.0_0.2_2_1000_500_20_rnn_False_1_4_tanh_300_0.2_1000_0.001_200.0.json",
        8: "cma_heterogeneous_individual_nn_slope_1298508492_8_16_1_4_8_16_1_3_7_1_3.0_0.2_2_1000_500_20_rnn_False_1_4_tanh_400_0.2_1000_0.001_200.0.json",
        10: "cma_heterogeneous_individual_nn_slope_878115724_10_20_1_4_8_20_1_3_7_1_3.0_0.2_2_1000_500_20_rnn_False_1_4_tanh_500_0.2_1000_0.001_200.0.json"}

    team_parameter_files = {
        2: "cma_heterogeneous_team_nn_slope_491264_2_4_1_4_8_4_1_3_7_1_3.0_0.2_2_1000_500_20_rnn_False_1_4_tanh_100_0.2_1000_0.001_200.0.json",
        4: "cma_heterogeneous_team_nn_slope_630311760_4_8_1_4_8_8_1_3_7_1_3.0_0.2_2_1000_500_20_rnn_False_1_4_tanh_100_0.2_1000_0.001_200.0.json",
        6: "cma_heterogeneous_team_nn_slope_4033523167_6_12_1_4_8_12_1_3_7_1_3.0_0.2_2_1000_500_20_rnn_False_1_4_tanh_120_0.2_1000_0.001_200.0.json",
        8: "cma_heterogeneous_team_nn_slope_1355129330_8_16_1_4_8_16_1_3_7_1_3.0_0.2_2_1000_500_20_rnn_False_1_4_tanh_160_0.2_1000_0.001_200.0.json",
        10: "cma_heterogeneous_team_nn_slope_1791095846_10_20_1_4_8_20_1_3_7_1_3.0_0.2_2_1000_500_20_rnn_False_1_4_tanh_100_0.2_1000_0.001_200.0.json"}

    ind_genomes = get_genomes(reward_level="individual", num_genomes=num_genomes)
    team_genomes = get_genomes(reward_level="team", num_genomes=num_genomes)

    default_num_episodes = 20
    f = None

    assert len(ind_genomes[2]) == len(team_genomes[2]), "Not testing the same number of genomes for ind and team setups"
    genomes_per_team = len(ind_genomes[2])

    if not os.path.exists(logfile_name):
        f = open(logfile_name, "w")
        header_string = ",".join([str(episode_num) for episode_num in range(samples_per_genome * genomes_per_team)])
        f.write(f"num_agents,reward_level,{header_string}\n")
    else:
        f = open(logfile_name, "a")

    for num_agents in range(2, 12, 2):
        # Get parameter and genome files
        parameter_filename = None
        top_genomes = None

        if reward_level == "individual":
            parameter_filename = ind_parameter_files[num_agents]
            top_genomes = ind_genomes

        elif reward_level == "team":
            parameter_filename = team_parameter_files[num_agents]
            top_genomes = team_genomes

        # Change number of episodes to the relevant number
        parameter_dictionary = json.loads(open(parameter_filename).read())
        new_num_episodes = int(samples_per_genome / num_agents)
        parameter_dictionary["environment"]["slope"]["num_episodes"] = new_num_episodes

        g = open(parameter_filename, "w")
        dictionary_string = json.dumps(parameter_dictionary, indent=4)
        g.write(dictionary_string)
        g.close()

        # Create fitness calculator
        fitness_calculator = FitnessCalculator(parameter_filename)

        # List of fitnesses for num_agents
        agent_fitnesses = []

        for index in range(len(top_genomes[num_agents])):
            genome = top_genomes[num_agents][index]

            # Make genome list
            genomes_list = []
            agents_list = []

            if reward_level == "individual":
                for i in range(num_agents):
                    genomes_list += [genome]
                    agents_list += [NNAgent(fitness_calculator.get_observation_size(), fitness_calculator.get_action_size(), parameter_filename, genomes_list[i])]

            elif reward_level == "team":
                full_genome = genome
                genome_part_length = int(len(full_genome) / num_agents)

                for i in range(num_agents):
                    genomes_list += [full_genome[i*genome_part_length : (i+1)*genome_part_length]]
                    agents_list += [NNAgent(fitness_calculator.get_observation_size(), fitness_calculator.get_action_size(), parameter_filename, genomes_list[i])]

            results = fitness_calculator.calculate_fitness(agents_list, render=False, time_delay=0)

            # Log fitness distribution
            for agent_scores in results["fitness_matrix"]:
                agent_fitnesses += agent_scores

        # Log to file
        std_string = ",".join([str(fitness) for fitness in agent_fitnesses])
        f.write(f"{num_agents},{reward_level},{std_string}\n")

        # Change number of episodes back
        parameter_dictionary = json.loads(open(parameter_filename).read())
        parameter_dictionary["environment"]["slope"]["num_episodes"] = default_num_episodes
        g = open(parameter_filename, "w")
        dictionary_string = json.dumps(parameter_dictionary, indent=4)
        g.write(dictionary_string)
        g.close()

    f.close()


def plot_agent_fitness_distribution(logfile_name, graph_file):
    bin_list = [i for i in range(-20000, 20000, 250)]
    #n_bins = 120
    data = pd.read_csv(logfile_name)
    num_reward_levels = 2
    num_agent_teams = 5
    fig1, ax1 = plt.subplots(num_reward_levels, num_agent_teams, figsize=(12, 6), sharex=True, sharey=True)
    #ax1.set_title('Fitness Distribution')
    fig1.suptitle('Fitness Distribution')

    plot_row = 0
    plot_col = 0

    for index, row in data.iterrows():
        # fitnesses = data.iloc[0, 2:]
        fitnesses = row[2:]

        ax1[plot_row][plot_col].set_ylabel(f'Frequency\n ({row["reward_level"].capitalize()})')
        ax1[plot_row][plot_col].set_xlabel(f'Fitness Value\n ({row["num_agents"]} agents)')
        ax1[plot_row][plot_col].hist(fitnesses, bins=bin_list)
        #ax1[plot_row][plot_col].hist(fitnesses, bins=n_bins)

        plot_col += 1
        if plot_col >= num_agent_teams:
            plot_row += 1
            plot_col = 0

    for ax in ax1.flat:
        ax.label_outer()

    plt.savefig(graph_file)


'''
# Get parameter file name
        "cma_heterogeneous_individual_nn_slope_550290314_4_8_1_4_8_8_1_3_7_1_3.0_0.2_2_1000_500_20_rnn_False_1_4_tanh_100_0.2_1000_0.001_200.0.json"
        parameter_files = glob(f'cma*json')
        parameter_filename = None
        
        for file in parameter_files:
            name = file.split("_")
            if name[6] == str(num_agents) and name[2] == reward_level:
                parameter_filename = file
                break
                

        # Get relevant genome
        "rwg_heterogeneous_individual_nn_slope_1_1_8_1_4_8_8_1_3_7_1_3.0_0.2_2_1000_500_20_rnn_False_1_4_tanh_1000_normal_0_1_-27.890000000000136.npy"
        genome_files = glob(f'rwg*npy')
        genome = None

        for genome_file in genome_files:
            name = genome_file.split("_")
            if name[]
'''

'''
all_parameter_files = get_all_files()
completed_files = get_completed_files()
unstarted_parameter_files = get_unstarted_files()
rerun_parameter_files = get_rerun_files(all_parameter_files, completed_files)
incomplete_files = get_incomplete_files(unstarted_parameter_files, rerun_parameter_files)
#copy_rerun_files(rerun_parameter_files)
#final_gen, y_vals, y_err_vals = get_generation_count(incomplete_files)

#for key in final_gen:
#    print(final_gen[key])

num_total = len(all_parameter_files)
num_completed = len(completed_files)
num_segfaults = len(count_seg_faults())
num_timeouts = len(count_timeouts())
num_unstarted = len(unstarted_parameter_files)
num_reruns = len(rerun_parameter_files)

print(f"Total: {num_total}")
print(f"Completed: {num_completed}")
print(f"Incomplete: {num_reruns}")
print(f"{num_completed + num_reruns} = {num_total}")

print(f"Segfaults: {num_segfaults}")
print(f"Timeouts: {num_timeouts}")
print(f"Missing json: {num_unstarted}")
print(f'{num_segfaults + num_timeouts + num_unstarted} = {num_reruns}')
'''

#plot_evolution_history("data/1_agent_arena_4/results", "evolution_history_1_agent_arena_4.png", num_generations=163, start_generation=1, step_size=1, y_min=-2000, y_max=35000)
#plot_evolution_history("data/4_agents/results", "evolution_history_4_agents_scaled.png", num_generations=109, start_generation=1, step_size=1, y_min=-2000, y_max=35000)
#plot_evolution_history_average("data/results", "evolution_history_2_agents.png", num_generations=1000, start_generation=20, step_size=20, y_min=-1000, y_max=80000)
#fix_results("data/results", 20, 1000, 20)

calculate_agent_fitness_distribution(reward_level="individual", num_genomes=30, samples_per_genome=120, logfile_name="agent_fitness_distribution.csv")
calculate_agent_fitness_distribution(reward_level="team", num_genomes=30, samples_per_genome=120, logfile_name="agent_fitness_distribution.csv")
plot_agent_fitness_distribution("agent_fitness_distribution.csv", "agent_fitness_distribution.png")

