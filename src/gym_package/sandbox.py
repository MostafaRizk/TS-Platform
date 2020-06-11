from scipy.stats import multivariate_normal
from scipy.stats import uniform
import numpy as np
import matplotlib.pyplot as plt

num_dimensions = 288
num_samples = 10
seed = 1
#random_variable = multivariate_normal(mean=None, cov=np.identity(num_dimensions))

min_array = np.full( (1, num_dimensions), -3)
max_array = np.full( (1, num_dimensions), 3)

random_state = np.random.RandomState(seed)

sampled_points = random_state.uniform(min_array, max_array, (num_samples, num_dimensions))

#samples = random_variable.rvs(10,1)

print(sampled_points)