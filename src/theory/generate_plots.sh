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

# Partial Cooperation True
# Generate team size vs cooperation data
#time python3 evolutionary_dynamics.py --parameters $folder_name/distribution_params_b_2_agents_alignment=0.9.json --function get_distribution_agents --args_to_pass team_size_vs_coop --num_agents 2
#time python3 evolutionary_dynamics.py --parameters $folder_name/distribution_params_b_2_agents_alignment=0.9.json --function get_distribution_agents --args_to_pass team_size_vs_coop --num_agents 4
#time python3 evolutionary_dynamics.py --parameters $folder_name/distribution_params_b_2_agents_alignment=0.9.json --function get_distribution_agents --args_to_pass team_size_vs_coop --num_agents 6
#time python3 evolutionary_dynamics.py --parameters $folder_name/distribution_params_b_2_agents_alignment=0.9.json --function get_distribution_agents --args_to_pass team_size_vs_coop --num_agents 8

# Plot team_size vs cooperation data
#time python3 evolutionary_dynamics.py --parameters $folder_name/distribution_params_b_2_agents_alignment=0.9.json --function plot_distribution_agents --args_to_pass team_size_vs_coop

# Partial Cooperation False
# Generate team size vs cooperation data
#time python3 evolutionary_dynamics.py --parameters $folder_name/distribution_params_c_2_agents_full_alignment_no_partial.json --function get_distribution_agents --args_to_pass team_size_vs_coop_no_partial --num_agents 2
#time python3 evolutionary_dynamics.py --parameters $folder_name/distribution_params_c_2_agents_full_alignment_no_partial.json --function get_distribution_agents --args_to_pass team_size_vs_coop_no_partial  --num_agents 4
#time python3 evolutionary_dynamics.py --parameters $folder_name/distribution_params_c_2_agents_full_alignment_no_partial.json --function get_distribution_agents --args_to_pass team_size_vs_coop_no_partial  --num_agents 6
#time python3 evolutionary_dynamics.py --parameters $folder_name/distribution_params_c_2_agents_full_alignment_no_partial.json --function get_distribution_agents --args_to_pass team_size_vs_coop_no_partial  --num_agents 8

# Plot team_size vs cooperation data
#time python3 evolutionary_dynamics.py --parameters $folder_name/distribution_params_c_2_agents_full_alignment_no_partial.json --function plot_distribution_agents --args_to_pass team_size_vs_coop_no_partial

# Partial Cooperation False (Less Novices)
# Generate team size vs cooperation data
#time python3 evolutionary_dynamics.py --parameters $folder_name/distribution_params_d_2_agents_full_alignment_no_partial_less_novices.json --function get_distribution_agents --args_to_pass team_size_vs_coop_no_partial_less_novices --num_agents 2
#time python3 evolutionary_dynamics.py --parameters $folder_name/distribution_params_d_2_agents_full_alignment_no_partial_less_novices.json --function get_distribution_agents --args_to_pass team_size_vs_coop_no_partial_less_novices  --num_agents 4
#time python3 evolutionary_dynamics.py --parameters $folder_name/distribution_params_d_2_agents_full_alignment_no_partial_less_novices.json --function get_distribution_agents --args_to_pass team_size_vs_coop_no_partial_less_novices  --num_agents 6
#time python3 evolutionary_dynamics.py --parameters $folder_name/distribution_params_d_2_agents_full_alignment_no_partial_less_novices.json --function get_distribution_agents --args_to_pass team_size_vs_coop_no_partial_less_novices  --num_agents 8

# Plot team_size vs cooperation data
#time python3 evolutionary_dynamics.py --parameters $folder_name/distribution_params_d_2_agents_full_alignment_no_partial_less_novices.json --function plot_distribution_agents --args_to_pass team_size_vs_coop_no_partial_less_novices

# Payoff vs Agents
#time python3 evolutionary_dynamics.py --parameters $folder_name/distribution_params_a_2_agents_full_alignment.json --function plot_payoff_vs_agents --args_to_pass 50
#time python3 evolutionary_dynamics.py --parameters $folder_name/distribution_params_b_2_agents_alignment=0.9.json --function plot_payoff_vs_agents --args_to_pass 50

# Price of anarchy (Fitness)
#time python3 evolutionary_dynamics.py --parameters $folder_name/price_of_anarchy_a_fitness.json --function calculate_price_of_anarchy --args_to_pass fitness --num_agents 2 --slope 8
#time python3 evolutionary_dynamics.py --parameters $folder_name/price_of_anarchy_a_fitness.json --function calculate_price_of_anarchy --args_to_pass fitness --num_agents 4 --slope 8
#time python3 evolutionary_dynamics.py --parameters $folder_name/price_of_anarchy_a_fitness.json --function calculate_price_of_anarchy --args_to_pass fitness --num_agents 6 --slope 8
#time python3 evolutionary_dynamics.py --parameters $folder_name/price_of_anarchy_a_fitness.json --function calculate_price_of_anarchy --args_to_pass fitness --num_agents 8 --slope 8

#time python3 evolutionary_dynamics.py --parameters $folder_name/price_of_anarchy_a_fitness.json --function plot_price_of_anarchy --args_to_pass fitness

# Price of anarchy (Payoff)
#time python3 evolutionary_dynamics.py --parameters $folder_name/price_of_anarchy_b_payoff.json --function calculate_price_of_anarchy --args_to_pass payoff --num_agents 2 --slope 8
#time python3 evolutionary_dynamics.py --parameters $folder_name/price_of_anarchy_b_payoff.json --function calculate_price_of_anarchy --args_to_pass payoff --num_agents 4 --slope 8
#time python3 evolutionary_dynamics.py --parameters $folder_name/price_of_anarchy_b_payoff.json --function calculate_price_of_anarchy --args_to_pass payoff --num_agents 6 --slope 8
#time python3 evolutionary_dynamics.py --parameters $folder_name/price_of_anarchy_b_payoff.json --function calculate_price_of_anarchy --args_to_pass payoff --num_agents 8 --slope 8
#time python3 evolutionary_dynamics.py --parameters $folder_name/price_of_anarchy_b_payoff.json --function calculate_price_of_anarchy --args_to_pass payoff --num_agents 10 --slope 8
#time python3 evolutionary_dynamics.py --parameters $folder_name/price_of_anarchy_b_payoff.json --function calculate_price_of_anarchy --args_to_pass payoff --num_agents 12 --slope 8

#slopes=(0 1 2 3 4 5 6 7 8)

#for i in ${!slopes[@]}
#do
#time python3 evolutionary_dynamics.py --parameters $folder_name/price_of_anarchy_b_payoff.json --function calculate_price_of_anarchy --args_to_pass payoff --num_agents 2 --slope ${slopes[$i]}
#done

#time python3 evolutionary_dynamics.py --parameters $folder_name/price_of_anarchy_b_payoff.json --function plot_price_of_anarchy --args_to_pass payoff

# Price of anarchy (No Partial)
#time python3 evolutionary_dynamics.py --parameters $folder_name/price_of_anarchy_c_payoff_no_partial.json --function calculate_price_of_anarchy --args_to_pass no_partial --num_agents 2 --slope 8
#time python3 evolutionary_dynamics.py --parameters $folder_name/price_of_anarchy_c_payoff_no_partial.json --function calculate_price_of_anarchy --args_to_pass no_partial --num_agents 4 --slope 8
#time python3 evolutionary_dynamics.py --parameters $folder_name/price_of_anarchy_c_payoff_no_partial.json --function calculate_price_of_anarchy --args_to_pass no_partial --num_agents 6 --slope 8
#time python3 evolutionary_dynamics.py --parameters $folder_name/price_of_anarchy_c_payoff_no_partial.json --function calculate_price_of_anarchy --args_to_pass no_partial --num_agents 8 --slope 8

#time python3 evolutionary_dynamics.py --parameters $folder_name/price_of_anarchy_c_payoff_no_partial.json --function plot_price_of_anarchy --args_to_pass no_partial





