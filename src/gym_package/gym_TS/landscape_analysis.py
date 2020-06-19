import numpy as np
import matplotlib.pyplot as plt

from pyDOE import *
from gym_TS.fitness_calculator import FitnessCalculator
from scipy.stats.distributions import norm
from scipy.stats import multivariate_normal

from gym_TS.agents.TinyAgent import TinyAgent

import pandas as pd


def generate_genomes(team_type, selection_level, distribution, num_samples):
    num_dimensions = 288
    seed = 1
    sampled_points = None

    # Sample using scipy's multivariate normal distribution implementation
    if distribution == "normal":
        mean_array = [0]*num_dimensions
        random_variable = multivariate_normal(mean=mean_array, cov=np.identity(num_dimensions)*3)
        sampled_points = random_variable.rvs(num_samples, seed)

    # Sample using numpy's uniform distribution implementation
    if distribution == "uniform":
        min_array = np.full((1, num_dimensions), -10)
        max_array = np.full((1, num_dimensions), 10)
        random_state = np.random.RandomState(seed)
        sampled_points = random_state.uniform(min_array, max_array, (num_samples, num_dimensions))

    # Do latin hypercube sampling to generate genomes (NN weights for agents)
    # sampled_points = lhs(num_dimensions, samples=num_samples) # [0,1] range

    # Transform points using cutoff
    #sampled_points = sampled_points*6 - 3  # [-3,3] range

    # Transform points using in-built transformation function
    #sampled_points = norm(loc=0, scale=1).ppf(sampled_points)

    # Calculate fitness of each genome
    fitness_calculator = FitnessCalculator(random_seed=seed, simulation_length=500,
                                                       num_trials=5, num_robots=2,
                                                       num_resources=3,
                                                       sensor_range=1, slope_angle=40,
                                                       arena_length=8, arena_width=4,
                                                       cache_start=1,
                                                       slope_start=3, source_start=7,
                                                       upward_cost_factor=3.0,
                                                       downward_cost_factor=0.2, carry_factor=2.0,
                                                       resource_reward_factor=1000.0,
                                                       using_gym=False)
    #python3 main.py --algorithm bootstrap --team_type homogeneous --selection_level team --generations 10000 --simulation_length 500 --trials 5  --seed 1 --num_robots 2 --num_resources 3 --sensor_range 1 --slope_angle 40 --arena_length 8 --arena_width 4 --cache_start 1 --slope_start 3 --source_start 7 --upward_cost_factor 3 --downward_cost_factor 0.2 --carry_factor 2 --resource_reward_factor 1000 --target_fitness 100

    ''''''
    f = open(f"genomes_{distribution}_{team_type}_{selection_level}.csv", "w")

    for individual in sampled_points:
        fitness_1, fitness_2 = fitness_calculator.calculate_fitness(individual_1=individual, individual_2=individual, render=False)
        team_fitness = fitness_1 + fitness_2

        genome_str = str(individual.tolist()).strip("[]") + "," + str(fitness_1) + "," + str(fitness_2) + "," + str(team_fitness) + "\n"
        f.write(genome_str)

    f.close()


def analyse_motion(team_type, selection_level, visualise, distribution):
    using_gym = False
    render = False
    time_delay = 0

    if visualise:
        using_gym = True
        render = True
        time_delay = 0.1

    f = open(f"genomes_{distribution}_{team_type}_{selection_level}.csv", "r")
    data = f.read().strip().split("\n")
    f.close()

    # Create fitness calculator
    fitness_calculator = FitnessCalculator(random_seed=1, simulation_length=500,
                                           num_trials=5, num_robots=2,
                                           num_resources=3,
                                           sensor_range=1, slope_angle=40,
                                           arena_length=8, arena_width=4,
                                           cache_start=1,
                                           slope_start=3, source_start=7,
                                           upward_cost_factor=3.0,
                                           downward_cost_factor=0.2, carry_factor=2.0,
                                           resource_reward_factor=1000.0,
                                           using_gym=using_gym)

    for row in data[0:30]:
        genome = np.array([float(element) for element in row.split(",")[0:-3]])
        fitness_1, fitness_2 = fitness_calculator.calculate_fitness_with_logging(individual_1=genome, individual_2=genome, render=render, time_delay=time_delay)
        team_fitness = fitness_1 + fitness_2
        print(team_fitness)


def plot_fitness_distribution(results_file, graph_file):
    """
    Plot distribution of random points from the fitness landscape as violin plots
    """
    # Read data
    #data = pd.read_csv(results_file)
    f = open(results_file, "r")
    data = f.read().strip().split("\n")
    f.close()


    # Format data
    results = {  # "Homogeneous-Individual": {"Individual 1": [], "Individual 2": []},
        "Homogeneous-Team": [],
    }

    data_points = 0

    for row in data:
        results["Homogeneous-Team"] += [float(row.split(",")[-1])]

    # Plot data
    fig1, ax1 = plt.subplots(figsize=(12, 4))
    ax1.set_title('Distribution of Landscape')
    ax1.set_ylim(-4000, 4000)
    ax1.set_ylabel('Fitness')
    ax1.set_xlabel('Evolutionary Configuration')

    positions = [0]
    # configs = ["Homogeneous-Individual", "Homogeneous-Team", "Heterogeneous-Individual", "Heterogeneous-Team"]
    # configs = ["Homogeneous-Team", "Heterogeneous-Individual"]
    configs = ["Homogeneous-Team"]

    for i in range(len(configs)):
        config = configs[i]
        ax1.violinplot(results[config], positions=[positions[i]], widths=0.3)

    ax1.set_xticklabels([x for x in configs])
    ax1.set_xticks([x for x in positions])

    #hB, = ax1.plot([1, 1], 'r-')
    #hB.set_visible(False)

    #plt.show()
    plt.savefig(graph_file)


def plot_action_distribution(genome_file, graph_file):
    num_dimensions = 288
    num_observations = 500*5
    num_networks = 500
    seed = 1

    observation_length = 41
    action_length = 6

    # Get samples
    f = open(genome_file, "r")
    data = f.read().strip().split("\n")
    f.close()

    # Sample num_observations observations uniformly
    min_array = np.full((1, observation_length), 0)
    max_array = np.full((1, observation_length), 2)
    random_state = np.random.RandomState(seed)
    # sampled_points = random_state.uniform(min_array, max_array, (num_samples, num_dimensions))
    observation_sequence = []

    for i in range(num_observations):
        new_observation = random_state.randint(min_array, max_array)
        observation_sequence += [new_observation]

    actions = [0, 0, 0, 0, 0, 0]

    # For all genomes
    for row in data:
        # Get genome
        genome = np.array([float(element) for element in row.split(",")[0:-3]])

        # Test genome on observation
        try:
            network = TinyAgent(observation_length, action_length, seed)
            network.load_weights(genome)
        except:
            genome = np.array([float(element) for element in row.split(",")])
            network = TinyAgent(observation_length, action_length, seed)
            network.load_weights(genome)

        for observation in observation_sequence:
            action = network.act(observation)
            actions[action] += 1

    # Plot data
    fig1, ax1 = plt.subplots(figsize=(12, 4))
    ax1.set_title('Most Chosen Action')
    ax1.set_ylabel('Times chosen')
    ax1.set_xlabel('Action')
    ax1.bar([0,1,2,3,4,5],actions)
    plt.savefig(graph_file)


def plot_weight_distribution(genome_file, graph_file):
    num_dimensions = 288

    # Get samples
    f = open(genome_file, "r")
    data = f.read().strip().split("\n")
    f.close()

    # Plot data
    fig1, ax1 = plt.subplots(figsize=(12, 4))
    ax1.set_title('Distribution of Neural Network Weights')
    #ax1.set_ylim(-1000, 1000)
    ax1.set_ylabel('Weight Value')
    ax1.set_xlabel('Weight ID')

    x_range = np.arange(num_dimensions)
    np_data = []

    for row in data:
        genome = np.array([float(element) for element in row.split(",")[0:-3]])
        np_data += [genome]

    y = np.transpose(np.array(np_data))

    ax1.boxplot(y)

    #plt.show()
    plt.savefig(graph_file)


def get_best_genomes(results_folder, results_file, team_type, selection_level, cutoff_score):
    """
    Load the genomes from a particular results folder that performed better than a certain cutoff score.
    Information on genome performance is based on a results file.
    """
    #Load contents of results file
    data = pd.read_csv(results_file)

    f = open(f"best_genomes_{team_type}_{selection_level}_{cutoff_score}.csv", "w")

    for index, row in data.iterrows():
        if row[" Team Type"] == team_type and row[" Selection Level"] == selection_level and float(row[" Evolved Fitness"]) > cutoff_score:
            filename = results_folder + row[" Log File"].strip(".log") + ".npy"
            individual = np.load(filename)
            genome_str = str(individual.tolist()).strip("[]") + "\n"
            f.write(genome_str)

    f.close()


def plot_weight_histogram(genome_file, graph_file):
    num_dimensions = 288

    # Get samples
    f = open(genome_file, "r")
    data = f.read().strip().split("\n")
    f.close()

    # Plot data
    fig1, ax1 = plt.subplots(figsize=(12, 4))
    ax1.set_title('Distribution of Neural Network Weights')
    # ax1.set_ylim(-1000, 1000)
    ax1.set_ylabel('Number of Weights')
    ax1.set_xlabel('Weight Magnitude')

    weights = {}
    min_weight = float("inf")
    max_weight = float("-inf")

    # For every weight in every genome
    for row in data:
        for element in row.split(",")[0:-3]:
            element_value = int(float(element))

            if element_value < min_weight:
                min_weight = element_value

            if element_value > max_weight:
                max_weight = element_value

            key = str(element_value)

            if key in weights:
                weights[key] += 1
            else:
                weights[key] = 1

    int_keys = [int(key) for key in weights.keys()]
    int_keys = sorted(int_keys)
    positions = [x for x in range(len(int_keys))]
    ax1.set_xticklabels([x for x in int_keys])
    ax1.set_xticks([x for x in positions])

    for i in range(len(int_keys)):
        key = int_keys[i]
        ax1.bar(positions[i], weights[str(key)])

    # plt.show()
    plt.savefig(graph_file)


def plot_activation_progression(genome_file, graph_file):
    """
    Plot distribution of activation values for each of the 6 outputs as violin plots
    """
    num_dimensions = 288
    num_observations = 500*5
    num_networks = 500
    seed = 1

    observation_length = 41
    action_length = 6

    # Get samples
    f = open(genome_file, "r")
    data = f.read().strip().split("\n")
    f.close()

    # Sample num_observations observations uniformly
    min_array = np.full((1, observation_length), 0)
    max_array = np.full((1, observation_length), 2)
    random_state = np.random.RandomState(seed)
    # sampled_points = random_state.uniform(min_array, max_array, (num_samples, num_dimensions))
    observation_sequence = []

    for i in range(num_observations):
        new_observation = random_state.randint(min_array, max_array)
        observation_sequence += [new_observation]

    activations = [[], [], [], [], [], []]

    # For all genomes
    for row in data[0:1]:
        # Get genome
        genome = np.array([float(element) for element in row.split(",")[0:-3]])

        # Test genome on observation
        try:
            network = TinyAgent(observation_length, action_length, seed)
            network.load_weights(genome)
        except:
            genome = np.array([float(element) for element in row.split(",")])
            network = TinyAgent(observation_length, action_length, seed)
            network.load_weights(genome)

        for observation in observation_sequence:
            network_outputs = network.get_all_activations(observation)

            for i in range(action_length):
                activations[i] += [network_outputs[i]]

    # Plot data
    fig1, ax1 = plt.subplots(figsize=(12, 4))

    positions = [0, 1, 2, 3, 4, 5]

    #for i in range(len(positions)):
    #    ax1.violinplot(activations[i], positions=[positions[i]], widths=0.3)

    timesteps = np.array( [x for x in range(num_observations)] )
    #ax1.errorbar(timesteps, activations)
    ax1.errorbar(timesteps, [abs(x) for x in activations[0]])
    ax1.errorbar(timesteps, [abs(x) for x in activations[1]])
    ax1.errorbar(timesteps, [abs(x) for x in activations[2]])
    ax1.errorbar(timesteps, [abs(x) for x in activations[3]])
    #ax1.plot(timesteps, [abs(x) for x in activations[4]])
    #ax1.plot(timesteps, [abs(x) for x in activations[5]])

    #for i in range(len(timesteps)):
        #print(f"{timesteps[i]}: {activations[0][i]} / {activations[1][i]} / {activations[2][i]} / {activations[3][i]} / {activations[4][i]} / {activations[5][i]} ==> {np.array([activations[x][i] for x in range(6)]).argmax()}")


    #ax1.set_xticklabels([x for x in positions])
    #ax1.set_xticks([x for x in positions])

    ax1.set_title('Activation Value Over Time')
    ax1.set_ylabel('Activation Value')
    ax1.set_xlabel('Number of Observations')
    plt.savefig(graph_file)


def plot_action_progression(genome_file, graph_file):
    """
    Plot distribution of activation values for each of the 6 outputs as violin plots
    """
    num_dimensions = 288
    num_observations = 500*5
    num_networks = 500
    seed = 1

    observation_length = 41
    action_length = 6

    # Get samples
    f = open(genome_file, "r")
    data = f.read().strip().split("\n")
    f.close()

    # Sample num_observations observations uniformly
    min_array = np.full((1, observation_length), 0)
    max_array = np.full((1, observation_length), 2)
    random_state = np.random.RandomState(seed)
    # sampled_points = random_state.uniform(min_array, max_array, (num_samples, num_dimensions))
    observation_sequence = []

    for i in range(num_observations):
        new_observation = random_state.randint(min_array, max_array)
        observation_sequence += [new_observation]

    action_list = [0] * num_observations
    action_at_time = [action_list[:] for x in range(action_length)]

    # For all genomes
    for row in data:
        # Get genome
        genome = np.array([float(element) for element in row.split(",")[0:-3]])

        # Test genome on observation
        try:
            network = TinyAgent(observation_length, action_length, seed)
            network.load_weights(genome)
        except:
            genome = np.array([float(element) for element in row.split(",")])
            network = TinyAgent(observation_length, action_length, seed)
            network.load_weights(genome)

        for i in range(len(observation_sequence)):
            observation = observation_sequence[i]
            action = network.act(observation)

            action_at_time[action][i] += 1

    # Plot data
    fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(6, 1, sharex=True, figsize=(12, 8))
    x_axis = [x for x in range(num_observations)]

    ''''''
    ax1.bar(x_axis, action_at_time[0])
    ax2.bar(x_axis, action_at_time[1])
    ax3.bar(x_axis, action_at_time[2])
    ax4.bar(x_axis, action_at_time[3])
    ax5.bar(x_axis, action_at_time[4])
    ax6.bar(x_axis, action_at_time[5])
    
    ax1.set_title('Frequency of Action Selection Over Time')

    ax1.set_ylabel('0')
    ax2.set_ylabel('1')
    ax3.set_ylabel('2')
    ax4.set_ylabel('3')
    ax5.set_ylabel('4')
    ax6.set_ylabel('5')

    ax6.set_xlabel('Observation Step')

    plt.savefig(graph_file)


distribution = "normal"
team_type = "homogeneous"
selection_level = "team"
num_samples = 10000
#generate_genomes(team_type, selection_level, distribution, num_samples)
#plot_fitness_distribution(f"genomes_{distribution}_{team_type}_{selection_level}.csv", f"fitness_distribution_{distribution}_{team_type}_{selection_level}.png")
#plot_weight_distribution(f"genomes_{distribution}_{team_type}_{selection_level}.csv", f"weight_distribution_{distribution}_{team_type}_{selection_level}.png")
#plot_weight_histogram(f"genomes_{distribution}_{team_type}_{selection_level}.csv",f"weight_histogram_{distribution}_{team_type}_{selection_level}.png")
#plot_action_distribution(f"genomes_{distribution}_{team_type}_{selection_level}.csv", f"action_distribution_{distribution}_{team_type}_{selection_level}.png")
#analyse_motion(team_type, selection_level, False, distribution)
#plot_activation_progression(f"genomes_{distribution}_{team_type}_{selection_level}.csv", f"activation_distribution_{distribution}_{team_type}_{selection_level}.png")
plot_action_progression(f"genomes_{distribution}_{team_type}_{selection_level}.csv", f"action_progression_{distribution}_{team_type}_{selection_level}.png")

#get_best_genomes("/Users/mostafa/Documents/Code/PhD/Results/Paper1/3_FixedTimeLag-e31c5b28867eeeb488fc051cbc4e3b09ce8beb31/", "results_sorted.csv", "homogeneous", "team", 0.0)
#plot_weight_histogram("best_genomes_homogeneous_team_0.0.csv", "best_genomes_homogeneous_team_0.0.png")
#plot_action_distribution("best_genomes_homogeneous_team_0.0.csv", "action_distribution_best_homogeneous_team_0.0.png")

#get_best_genomes("/Users/mostafa/Documents/Code/PhD/Results/Paper1/3_FixedTimeLag-e31c5b28867eeeb488fc051cbc4e3b09ce8beb31/", "results_sorted.csv", "heterogeneous", "individual", 0.0)
#plot_weight_histogram("best_genomes_heterogeneous_individual_0.0.csv", "best_genomes_heterogeneous_individual_0.0.png")
#plot_action_distribution("best_genomes_heterogeneous_individual_0.0.csv", "action_distribution_best_heterogeneous_individual_0.0.png")


