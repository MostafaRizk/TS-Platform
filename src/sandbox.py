import numpy as np
from fitness import FitnessCalculator
from scipy.stats import multivariate_normal
from agents.NNAgent import NNAgent

seed = 1
num_dimensions = 288
num_samples = 2

fitness_calculator = FitnessCalculator('default_parameters.json')


mean_array = [0]*num_dimensions
random_variable = multivariate_normal(mean=mean_array, cov=np.identity(num_dimensions)*3)
sampled_points = random_variable.rvs(num_samples, seed)
genome = sampled_points[0]

agent1 = NNAgent(fitness_calculator.get_observation_size(), fitness_calculator.get_action_size(), 'default_parameters.json')
agent1.load_weights(genome)
agent2 = NNAgent(fitness_calculator.get_observation_size(), fitness_calculator.get_action_size(), 'default_parameters.json')
agent2.load_weights(genome)

fitness_data = fitness_calculator.calculate_fitness(agent1, agent2, render=False, time_delay=0, measure_specialisation=False, logging=True, logfilename="demo_action_log.csv")
fitness_1 = fitness_data["fitness_1"]
fitness_2 = fitness_data["fitness_2"]
specialisation = fitness_data["specialisation"]


print(f"Fitness 1: {fitness_1} Fitness 2: {fitness_2} Specialisation {specialisation}")