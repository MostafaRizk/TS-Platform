import numpy as np


def calculate_distance(metric, a, b):
    a = np.array(a)
    b = np.array(b)
    if metric == "euclidean":
        return np.linalg.norm(a - b)