#!/bin/bash

folder_name=/Users/mostafa/Documents/Code/PhD/TS-Platform/results/2022_09_14_replicator_mutator
rwg_file=$folder_name/all_genomes_centralised_rwg_heterogeneous_team_nn_slope_1791095846_2_4_1_8_8_4_1_3_7_1_3.0_0.2_2_1000_100_20_False_rnn_False_1_4_tanh_20000_normal_0_1_.csv
param_file=$folder_name/CTDE_rnn_False_1.json
traj_file=$folder_name/trajectories.csv
reduction_file=$folder_name/reduction.npy
plot_3D_file=$folder_name/plot_3D.pdf
cluster_prefix=$folder_name/kmeans
#centroid_plots=$folder_name/centroids.pdf

# Plot trend
#time python3 evolutionary_dynamics.py --parameters $folder_name/mutation_2_distance_0.999_nonovice.json --function trend --args_to_pass [0.91,0.03,0.03,0.03]

# Plot simplex
#time python3 evolutionary_dynamics.py --parameters $folder_name/mutation_2_distance.json --function plot_simplex
#time python3 evolutionary_dynamics.py --parameters $folder_name/mutation_2_distance_equal_probs.json --function plot_simplex
#time python3 evolutionary_dynamics.py --parameters $folder_name/mutation_2_distance_0.99.json --function plot_simplex
#time python3 evolutionary_dynamics.py --parameters $folder_name/mutation_2_distance_0.999_nonovice.json --function plot_simplex
#time python3 evolutionary_dynamics.py --parameters $folder_name/distribution_params_b_2_agents_alignment=0.9_no_mutation.json --function plot_simplex
#time python3 evolutionary_dynamics.py --parameters $folder_name/mutation_2_distance_0.999_nonovice_debugging.json --function plot_simplex
#time python3 evolutionary_dynamics.py --parameters $folder_name/mutation_2_distance_0.9_nonovice_debugging.json --function plot_simplex

# Dimensionality reduction
#time python3 dimensionality.py --generate_trajectories true --rwg_genomes_file $rwg_file --parameter_path $param_file --save_file $traj_file
#time python3 dimensionality.py --perform_reduction true --trajectories_file $traj_file --save_file $reduction_file
#time python3 dimensionality.py --plot_3D true --reduction_file $reduction_file --trajectories_file $traj_file --save_file $plot_3D_file
#time python3 dimensionality.py --clustering true --trajectories_file $traj_file --save_file_prefix $cluster_prefix
#time python3 dimensionality.py --plot_centroids true --data_file_prefix $cluster_prefix --save_file_prefix $cluster_prefix

# Plot alignment
#time python3 evolutionary_dynamics.py --parameters $folder_name/mutation_2_distance_0.999_nonovice.json --function plot_alignment_frequency
#time python3 evolutionary_dynamics.py --parameters $folder_name/mutation_2_distance_0.999_nonovice.json --function plot_alignment_frequency
#time python3 evolutionary_dynamics.py --parameters $folder_name/distribution_params_b_2_agents_alignment=0.9_no_mutation.json --function plot_alignment_frequency
time python3 evolutionary_dynamics.py --parameters $folder_name/mutation_2_distance_0.999_nonovice_debugging.json --function plot_alignment_frequency






