import numpy as np
from fitness import FitnessCalculator
from scipy.stats import multivariate_normal

seed = 1
num_dimensions = 288
num_samples = 2

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


mean_array = [0]*num_dimensions
random_variable = multivariate_normal(mean=mean_array, cov=np.identity(num_dimensions)*3)
sampled_points = random_variable.rvs(num_samples, seed)
individual = sampled_points[0]
fitness_1, fitness_2 = fitness_calculator.calculate_fitness(individual_1=individual, individual_2=individual, render=True)