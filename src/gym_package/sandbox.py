from scipy.stats import multivariate_normal
import numpy as np
import matplotlib.pyplot as plt

num_dimensions = 288
random_variable = multivariate_normal(mean=None, cov=np.identity(num_dimensions))

samples = random_variable.rvs(10,1)

print(len(samples))