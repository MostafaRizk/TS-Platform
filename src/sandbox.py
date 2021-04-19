import numpy as np

from scipy.stats import dirichlet
from scipy.optimize import linprog


def sample_distributions(num_samples=1000):
    num_strategies = 4
    sample_format = np.ones(num_strategies)
    return dirichlet.rvs(size=num_samples, alpha=sample_format)


x0 = sample_distributions(1)[0]

