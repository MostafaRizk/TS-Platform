import numpy as np
import subprocess
import os

generator_seed = 2
num_seeds = 30
path = "results/2021_10_06_b_aamas_data/2021_10_04_scalability_constant_arena_correct_neurons/data/"
np_random = np.random.RandomState(generator_seed)
#os.chdir(path)

array_string = ""

for i in range(30):
    new_seed = np_random.randint(low=1, high=2 ** 32 - 1)
    if i != 0:
        array_string += " "
    array_string += f"\"{new_seed}\""

print(array_string)