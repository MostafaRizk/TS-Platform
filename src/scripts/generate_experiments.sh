#!/bin/bash

#python3 experiment_generator_cma.py --experiment_directory /Users/mostafa/Documents/Code/PhD/TS-Platform/results/2021_10_01_fc_many_neurons --core_parameter_filename /Users/mostafa/Documents/Code/PhD/TS-Platform/src/default_slope_parameters.json --learning_type fully-centralised --reward_level team --list_file_name LIST_fully-centralised --num_agents_in_setup [2,4,6,8] --holding_constant games_per_learner --constant_arena_size False --num_seeds_for_team 30 --generator_seed 1
#python3 experiment_generator_cma.py --experiment_directory /Users/mostafa/Documents/Code/PhD/TS-Platform/results/2021_10_01_fc_many_neurons_constant_arena --core_parameter_filename /Users/mostafa/Documents/Code/PhD/TS-Platform/src/default_slope_parameters.json --learning_type fully-centralised --reward_level team --list_file_name LIST_fully-centralised --num_agents_in_setup [2,4,6,8] --holding_constant games_per_learner --constant_arena_size True --num_seeds_for_team 30 --generator_seed 1
#python3 experiment_generator_cma.py --experiment_directory /Users/mostafa/Documents/Code/PhD/TS-Platform/results/2021_10_01_fc_many_neurons_constant_arena --core_parameter_filename /Users/mostafa/Documents/Code/PhD/TS-Platform/src/default_slope_parameters.json --learning_type fully-centralised --reward_level team --list_file_name LIST_fully-centralised --num_agents_in_setup [2,4,6,8] --holding_constant games_per_learner --constant_arena_size True --num_seeds_for_team 30 --generator_seed 2
#python3 experiment_generator_cma.py --experiment_directory /Users/mostafa/Documents/Code/PhD/TS-Platform/results/2021_10_04_fc_variable_arena_bigger_pop_more_neurons --core_parameter_filename /Users/mostafa/Documents/Code/PhD/TS-Platform/src/default_slope_parameters.json --learning_type fully-centralised --reward_level team --list_file_name LIST_fully-centralised --num_agents_in_setup [2,4,6,8] --holding_constant games_per_run_2 --constant_arena_size False --num_seeds_for_team 30 --generator_seed 1
#python3 experiment_generator_cma.py --experiment_directory /Users/mostafa/Documents/Code/PhD/TS-Platform/results/2021_10_06_b_aamas_data/2021_11_29_ctde_variable_arena_bigger_pop_more_neurons --core_parameter_filename /Users/mostafa/Documents/Code/PhD/TS-Platform/src/default_slope_parameters.json --learning_type centralised --reward_level team --list_file_name LIST_ctde --num_agents_in_setup [2,4,6,8] --holding_constant games_per_run_2 --constant_arena_size False --num_seeds_for_team 30 --generator_seed 1

#python3 experiment_generator_cma.py --experiment_directory /Users/mostafa/Documents/Code/PhD/TS-Platform/results/2021_10_06_b_aamas_data/2021_12_15_scalability_variable_arena_10_agents --core_parameter_filename /Users/mostafa/Documents/Code/PhD/TS-Platform/src/default_slope_parameters.json --learning_type fully-centralised --reward_level team --list_file_name LIST_fc --num_agents_in_setup [10] --holding_constant games_per_learner --constant_arena_size False --num_seeds_for_team 30 --generator_seed 1
#python3 experiment_generator_cma.py --experiment_directory /Users/mostafa/Documents/Code/PhD/TS-Platform/results/2021_10_06_b_aamas_data/2021_12_15_scalability_variable_arena_10_agents --core_parameter_filename /Users/mostafa/Documents/Code/PhD/TS-Platform/src/default_slope_parameters.json --learning_type centralised --reward_level team --list_file_name LIST_ctde --num_agents_in_setup [10] --holding_constant games_per_learner --constant_arena_size False --num_seeds_for_team 30 --generator_seed 1
#python3 experiment_generator_cma.py --experiment_directory /Users/mostafa/Documents/Code/PhD/TS-Platform/results/2021_10_06_b_aamas_data/2021_12_15_scalability_variable_arena_10_agents --core_parameter_filename /Users/mostafa/Documents/Code/PhD/TS-Platform/src/default_slope_parameters.json --learning_type decentralised --reward_level individual --list_file_name LIST_dec --num_agents_in_setup [10] --holding_constant games_per_learner --constant_arena_size False --num_seeds_for_team 30 --generator_seed 1
#python3 experiment_generator_cma.py --experiment_directory /Users/mostafa/Documents/Code/PhD/TS-Platform/results/2021_10_06_b_aamas_data/2021_12_15_scalability_constant_arena_10_agents --core_parameter_filename /Users/mostafa/Documents/Code/PhD/TS-Platform/src/default_slope_parameters.json --learning_type fully-centralised --reward_level team --list_file_name LIST_fc_constant --num_agents_in_setup [10] --holding_constant games_per_learner --constant_arena_size False --num_seeds_for_team 30 --generator_seed 1
#python3 experiment_generator_cma.py --experiment_directory /Users/mostafa/Documents/Code/PhD/TS-Platform/results/2021_10_06_b_aamas_data/2021_12_15_scalability_constant_arena_10_agents --core_parameter_filename /Users/mostafa/Documents/Code/PhD/TS-Platform/src/default_slope_parameters.json --learning_type centralised --reward_level team --list_file_name LIST_ctde_constant --num_agents_in_setup [10] --holding_constant games_per_learner --constant_arena_size False --num_seeds_for_team 30 --generator_seed 1
#python3 experiment_generator_cma.py --experiment_directory /Users/mostafa/Documents/Code/PhD/TS-Platform/results/2021_10_06_b_aamas_data/2021_12_15_scalability_constant_arena_10_agents --core_parameter_filename /Users/mostafa/Documents/Code/PhD/TS-Platform/src/default_slope_parameters.json --learning_type decentralised --reward_level individual --list_file_name LIST_dec_constant --num_agents_in_setup [10] --holding_constant games_per_learner --constant_arena_size False --num_seeds_for_team 30 --generator_seed 1


#python3 experiment_generator_rwg.py --experiment_directory /Users/mostafa/Documents/Code/PhD/TS-Platform/results/2021_10_06_b_aamas_data/2021_10_09_rwg_ctde_improved  --core_parameter_filename /Users/mostafa/Documents/Code/PhD/TS-Platform/src/default_slope_rwg_analysis_parameters.json --learning_type centralised --reward_level team --list_file_name LIST_CTDE_rwg --num_agents_in_setup [2] --num_seeds_for_team 1 --generator_seed 1 --many_architectures True
#python3 experiment_generator_rwg.py --experiment_directory /Users/mostafa/Documents/Code/PhD/TS-Platform/results/2021_10_06_b_aamas_data/2021_10_09_rwg_ctde_improved --core_parameter_filename /Users/mostafa/Documents/Code/PhD/TS-Platform/src/default_slope_rwg_analysis_parameters.json --learning_type fully-centralised --reward_level team --list_file_name LIST_FC_rwg --num_agents_in_setup [2] --num_seeds_for_team 1 --generator_seed 1 --many_architectures True
#python3 experiment_generator_rwg.py --experiment_directory /Users/mostafa/Documents/Code/PhD/TS-Platform/results/2021_10_06_b_aamas_data/dummy_rwg --core_parameter_filename /Users/mostafa/Documents/Code/PhD/TS-Platform/results/2021_10_06_b_aamas_data/dummy_rwg_params.json --learning_type centralised --reward_level team --list_file_name LIST_CTDE_rwg --num_agents_in_setup [2] --num_seeds_for_team 1 --generator_seed 1 --many_architectures True
#python3 experiment_generator_rwg.py --experiment_directory /Users/mostafa/Documents/Code/PhD/TS-Platform/results/2021_10_06_b_aamas_data/2021_12_13_ctde_incremental  --core_parameter_filename /Users/mostafa/Documents/Code/PhD/TS-Platform/src/default_slope_rwg_analysis_parameters_incremental.json --learning_type centralised --reward_level team --list_file_name LIST_CTDE_rwg_incremental --num_agents_in_setup [2] --num_seeds_for_team 1 --generator_seed 1 --many_architectures True

#python3 experiment_generator_rwg.py --experiment_directory /Users/mostafa/Documents/Code/PhD/TS-Platform/results/2021_10_06_b_aamas_data/2021_10_27_a_rwg_ctde_0_slope  --core_parameter_filename /Users/mostafa/Documents/Code/PhD/TS-Platform/src/default_slope_rwg_analysis_parameters_flat.json --learning_type centralised --reward_level team --list_file_name LIST_CTDE_rwg_flat --num_agents_in_setup [2] --num_seeds_for_team 1 --generator_seed 1 --many_architectures True
#python3 experiment_generator_rwg.py --experiment_directory /Users/mostafa/Documents/Code/PhD/TS-Platform/results/2021_10_06_b_aamas_data/2021_10_27_b_rwg_ctde_lhs_0_slope  --core_parameter_filename /Users/mostafa/Documents/Code/PhD/TS-Platform/src/default_slope_lhs_parameters_flat.json --learning_type centralised --reward_level team --list_file_name LIST_CTDE_rwg_lhs_flat --num_agents_in_setup [2] --num_seeds_for_team 1 --generator_seed 1 --many_architectures False
#python3 experiment_generator_rwg.py --experiment_directory /Users/mostafa/Documents/Code/PhD/TS-Platform/results/2021_10_06_b_aamas_data/2021_10_27_c_rwg_fc_8_slope  --core_parameter_filename /Users/mostafa/Documents/Code/PhD/TS-Platform/src/default_slope_rwg_analysis_parameters.json --learning_type fully-centralised --reward_level team --list_file_name LIST_FC_rwg --num_agents_in_setup [2] --num_seeds_for_team 1 --generator_seed 1 --many_architectures True
#python3 experiment_generator_rwg.py --experiment_directory /Users/mostafa/Documents/Code/PhD/TS-Platform/results/2021_10_06_b_aamas_data/2021_10_27_d_rwg_fc_0_slope  --core_parameter_filename /Users/mostafa/Documents/Code/PhD/TS-Platform/src/default_slope_rwg_analysis_parameters_flat.json --learning_type fully-centralised --reward_level team --list_file_name LIST_FC_rwg_flat --num_agents_in_setup [2] --num_seeds_for_team 1 --generator_seed 1 --many_architectures True

#python3 experiment_generator_cma.py --experiment_directory /Users/mostafa/Documents/Code/PhD/TS-Platform/results/2021_10_06_b_aamas_data/2022_01_06_ctde_2_flat --core_parameter_filename /Users/mostafa/Documents/Code/PhD/TS-Platform/src/default_slope_parameters_flat_learning.json --learning_type centralised --reward_level team --list_file_name LIST_CTDE_flat --num_agents_in_setup [2] --holding_constant games_per_learner --constant_arena_size False --num_seeds_for_team 30 --generator_seed 1

python3 dimensionality_reduction.py --rwg_genomes_file /Users/mostafa/Documents/Code/PhD/TS-Platform/results/2021_10_06_b_aamas_data/2021_10_25_rwg_ctde_lhs/data/sorted_samples.csv --evolved_genomes_directory /Users/mostafa/Documents/Code/PhD/TS-Platform/results/2021_10_06_b_aamas_data/2022_01_06_ctde_2_flat/data/ctde_flat --results_file /Users/mostafa/Documents/Code/PhD/TS-Platform/results/2021_10_06_b_aamas_data/2022_01_06_ctde_2_flat/data/DR_slope.csv --num_episodes 30 --plot_destination /Users/mostafa/Documents/Code/PhD/TS-Platform/results/2021_10_06_b_aamas_data/2022_01_06_ctde_2_flat/new_DR.pdf