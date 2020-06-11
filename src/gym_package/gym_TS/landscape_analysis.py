import numpy as np
import matplotlib.pyplot as plt

from pyDOE import *
from gym_TS.fitness_calculator import FitnessCalculator
from scipy.stats.distributions import norm
from scipy.stats import multivariate_normal

from gym_TS.agents.TinyAgent import TinyAgent


def generate_genomes(team_type, selection_level, distribution):
    num_dimensions = 288
    num_samples = 500
    seed=1
    sampled_points = None

    # Sample using scipy's multivariate normal distribution implementation
    if distribution == "normal":
        mean_array = np.full((1, num_dimensions), 1)
        random_variable = multivariate_normal(mean=mean_array, cov=np.identity(num_dimensions))
        sampled_points = random_variable.rvs(num_samples, seed)

    # Sample using numpy's uniform distribution implementation
    if distribution == "uniform":
        min_array = np.full((1, num_dimensions), -3)
        max_array = np.full((1, num_dimensions), 3)
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

        genome_str = str(individual.tolist()).strip("[]") + "," + str(team_fitness) + "\n"
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

    f = open(f"flacco_{distribution}_{team_type}_{selection_level}.csv", "r")
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
        genome = np.array([float(element) for element in row.split(",")[0:-1]])
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
    ax1.set_ylim(-2000, 2000)
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
        genome = np.array([float(element) for element in row.split(",")[0:-1]])

        # Test genome on observation
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
        genome = np.array([float(element) for element in row.split(",")[0:-1]])
        np_data += [genome]

    y = np.transpose(np.array(np_data))

    ax1.boxplot(y)

    #plt.show()
    plt.savefig(graph_file)


generate_genomes(team_type="homogeneous", selection_level="team", distribution="normal")


