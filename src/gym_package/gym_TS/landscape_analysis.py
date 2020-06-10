import numpy as np
import matplotlib.pyplot as plt

from pyDOE import *
from gym_TS.fitness_calculator import FitnessCalculator
from scipy.stats.distributions import norm
from scipy.stats import multivariate_normal


def generate_sample(team_type, selection_level):
    num_dimensions = 288
    num_samples = 500
    seed=1

    random_variable = multivariate_normal(mean=None, cov=np.identity(num_dimensions))

    sampled_points = random_variable.rvs(num_samples, seed)

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
    f = open(f"flacco_gaussian_proper_{team_type}_{selection_level}.csv", "w")

    for individual in sampled_points:
        fitness_1, fitness_2 = fitness_calculator.calculate_fitness(individual_1=individual, individual_2=individual, render=False)
        team_fitness = fitness_1 + fitness_2

        genome_str = str(individual.tolist()).strip("[]") + "," + str(team_fitness) + "\n"
        f.write(genome_str)

    f.close()


def analyse_motion(team_type, selection_level, visualise):
    using_gym = False
    render = False
    time_delay = 0

    if visualise:
        using_gym = True
        render = True
        time_delay = 0.1

    f = open(f"flacco_gaussian_proper_{team_type}_{selection_level}.csv", "r")
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

#generate_lhs_sample(team_type="heterogeneous", selection_level="individual")

#genome = np.load("CMA_homogeneous_team_500_5000_5_17_2_3_1_40_8_4_1_3_7_3.0_0.2_2.0_1000.0_0.2_40.npy")
#print(genome)


#generate_sample(team_type="homogeneous", selection_level="team")
analyse_motion("homogeneous", "team", visualise=False)


