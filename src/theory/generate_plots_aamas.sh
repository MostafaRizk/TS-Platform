#!/bin/bash

folder_name=/Users/mostafa/Documents/Code/PhD/TS-Platform/results/2022_09_14_replicator_mutator

# Plot simplex
#time python3 evolutionary_dynamics.py --parameters $folder_name/distribution_params_b_2_agents_alignment=0.9.json --function plot_simplex
time python3 evolutionary_dynamics.py --parameters $folder_name/distribution_params_b_2_agents_alignment=0.9_some_mutation.json --function plot_simplex






