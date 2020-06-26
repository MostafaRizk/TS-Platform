from scipy.stats import multivariate_normal
from scipy.stats import uniform
import numpy as np
import matplotlib.pyplot as plt
import os

filename_1 = "genomes_normal_homogeneous_team.csv"
filename_2 = "genomes_normal_homogeneous_team_20001.csv"

parent = '/Users/mostafa/Documents/Code/PhD/TS-Platform/src/gym_package/gym_TS/'

f2 = open(parent+filename_2, "r")
contents = f2.read()
f2.close()

f1 = open(parent+filename_1,"a+")
f1.write(contents)
f1.close()