import numpy as np


def calculate_distance(metric, a, b):
    if metric == "euclidean":
        return np.linalg.norm(a - b)