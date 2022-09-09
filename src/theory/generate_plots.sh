#!/bin/bash

folder_name=/Users/mostafa/Documents/Code/PhD/TS-Platform/results/2021_11_09_a_game_theory/

# Trend plots
#time python3 evolutionary_dynamics.py --parameters $folder_name/trend_params.json --function trend --args_to_pass [0.91,0.03,0.03,0.03]
#time python3 evolutionary_dynamics.py --parameters $folder_name/trend_params.json --function trend --args_to_pass [0.91,0.06,0.015,0.015]

# Distribution with alignment probability 1.0
#time python3 evolutionary_dynamics.py --parameters $folder_name/distribution_params_a_2_agents_full_alignment.json --function get_distribution --args_to_pass distribution_data_a_2_agents_full_alignment.json
#time python3 evolutionary_dynamics.py --parameters $folder_name/distribution_params_a_2_agents_full_alignment.json --function plot_distribution --args_to_pass distribution_data_a_2_agents_full_alignment.json

# Distribution with alignment probability 0.9 (unused)
#time python3 evolutionary_dynamics.py --parameters $folder_name/distribution_params_b_2_agents_alignment=0.9.json --function get_distribution --args_to_pass distribution_data_b_2_agents_alignment=0.9.json
#time python3 evolutionary_dynamics.py --parameters $folder_name/distribution_params_b_2_agents_alignment=0.9.json --function plot_distribution --args_to_pass distribution_data_b_2_agents_alignment=0.9.json

# Alignment plot
# time python3 evolutionary_dynamics.py --parameters $folder_name/distribution_params_a_2_agents_full_alignment.json --function plot_alignment_frequency

# Payoff vs Agents
#time python3 evolutionary_dynamics.py --parameters $folder_name/distribution_params_a_2_agents_full_alignment.json --function plot_payoff_vs_agents --args_to_pass 50
#time python3 evolutionary_dynamics.py --parameters $folder_name/distribution_params_b_2_agents_alignment=0.9.json --function plot_payoff_vs_agents --args_to_pass 50
#time python3 evolutionary_dynamics.py --parameters $folder_name/distribution_params_a_2_agents_full_alignment.json --function plot_payoff_vs_agents_combined --args_to_pass 50 --alignments [0.0,0.1,0.5,0.9,1.0]

# Plot simplex
#time python3 evolutionary_dynamics.py --parameters $folder_name/distribution_params_b_2_agents_alignment=0.9.json --function plot_simplex
#time python3 evolutionary_dynamics.py --parameters $folder_name/distribution_params_e_2_agents_alignment=0.9_slope=4.json --function plot_simplex


# Partial Cooperation True
# Plot team_size vs cooperation data
time python3 evolutionary_dynamics.py --parameters $folder_name/distribution_params_b_2_agents_alignment=0.9.json --function plot_distribution_agents --args_to_pass team_size_vs_coop
#Slopes
#time python3 evolutionary_dynamics.py --parameters $folder_name/distribution_params_b_2_agents_alignment=0.9.json --function get_distribution_slopes --args_to_pass team_size_vs_coop --slope 0
#time python3 evolutionary_dynamics.py --parameters $folder_name/distribution_params_b_2_agents_alignment=0.9.json --function get_distribution_slopes --args_to_pass team_size_vs_coop --slope 1
#time python3 evolutionary_dynamics.py --parameters $folder_name/distribution_params_b_2_agents_alignment=0.9.json --function get_distribution_slopes --args_to_pass team_size_vs_coop --slope 2
#time python3 evolutionary_dynamics.py --parameters $folder_name/distribution_params_b_2_agents_alignment=0.9.json --function get_distribution_slopes --args_to_pass team_size_vs_coop --slope 3
#time python3 evolutionary_dynamics.py --parameters $folder_name/distribution_params_b_2_agents_alignment=0.9.json --function get_distribution_slopes --args_to_pass team_size_vs_coop --slope 4
#time python3 evolutionary_dynamics.py --parameters $folder_name/distribution_params_b_2_agents_alignment=0.9.json --function get_distribution_slopes --args_to_pass team_size_vs_coop --slope 5
#time python3 evolutionary_dynamics.py --parameters $folder_name/distribution_params_b_2_agents_alignment=0.9.json --function get_distribution_slopes --args_to_pass team_size_vs_coop --slope 6
#time python3 evolutionary_dynamics.py --parameters $folder_name/distribution_params_b_2_agents_alignment=0.9.json --function get_distribution_slopes --args_to_pass team_size_vs_coop --slope 7
#time python3 evolutionary_dynamics.py --parameters $folder_name/distribution_params_b_2_agents_alignment=0.9.json --function get_distribution_slopes --args_to_pass team_size_vs_coop --slope 8
#time python3 evolutionary_dynamics.py --parameters $folder_name/distribution_params_b_2_agents_alignment=0.9.json --function plot_distribution_slopes --args_to_pass team_size_vs_coop

# Partial Cooperation False
# Plot team_size vs cooperation data
#time python3 evolutionary_dynamics.py --parameters $folder_name/distribution_params_c_2_agents_full_alignment_no_partial.json --function plot_distribution_agents --args_to_pass team_size_vs_coop_no_partial

# Partial Cooperation False (Less Novices)
# Plot team_size vs cooperation data
#time python3 evolutionary_dynamics.py --parameters $folder_name/distribution_params_d_2_agents_full_alignment_no_partial_less_novices.json --function plot_distribution_agents --args_to_pass team_size_vs_coop_no_partial_less_novices

# Price of anarchy (Fitness)
#time python3 evolutionary_dynamics.py --parameters $folder_name/price_of_anarchy_a_fitness.json --function plot_price_of_anarchy --args_to_pass fitness

# Price of anarchy (Payoff)
#time python3 evolutionary_dynamics.py --parameters $folder_name/price_of_anarchy_b_payoff.json --function plot_price_of_anarchy --args_to_pass payoff

# Price of anarchy (No Partial)
#time python3 evolutionary_dynamics.py --parameters $folder_name/price_of_anarchy_c_payoff_no_partial.json --function plot_price_of_anarchy --args_to_pass no_partial







