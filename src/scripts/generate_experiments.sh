#!/bin/bash

#python3 experiment_generator_cma.py --experiment_directory /Users/mostafa/Documents/Code/PhD/TS-Platform/results/2021_10_01_fc_many_neurons --core_parameter_filename /Users/mostafa/Documents/Code/PhD/TS-Platform/src/default_slope_parameters.json --learning_type fully-centralised --reward_level team --list_file_name LIST_fully-centralised --num_agents_in_setup [2,4,6,8] --holding_constant games_per_learner --constant_arena_size False --num_seeds_for_team 30 --generator_seed 1
#python3 experiment_generator_cma.py --experiment_directory /Users/mostafa/Documents/Code/PhD/TS-Platform/results/2021_10_01_fc_many_neurons_constant_arena --core_parameter_filename /Users/mostafa/Documents/Code/PhD/TS-Platform/src/default_slope_parameters.json --learning_type fully-centralised --reward_level team --list_file_name LIST_fully-centralised --num_agents_in_setup [2,4,6,8] --holding_constant games_per_learner --constant_arena_size True --num_seeds_for_team 30 --generator_seed 1
#python3 experiment_generator_cma.py --experiment_directory /Users/mostafa/Documents/Code/PhD/TS-Platform/results/2021_10_01_fc_many_neurons_constant_arena --core_parameter_filename /Users/mostafa/Documents/Code/PhD/TS-Platform/src/default_slope_parameters.json --learning_type fully-centralised --reward_level team --list_file_name LIST_fully-centralised --num_agents_in_setup [2,4,6,8] --holding_constant games_per_learner --constant_arena_size True --num_seeds_for_team 30 --generator_seed 2
#python3 experiment_generator_cma.py --experiment_directory /Users/mostafa/Documents/Code/PhD/TS-Platform/results/2021_10_04_fc_variable_arena_bigger_pop_more_neurons --core_parameter_filename /Users/mostafa/Documents/Code/PhD/TS-Platform/src/default_slope_parameters.json --learning_type fully-centralised --reward_level team --list_file_name LIST_fully-centralised --num_agents_in_setup [2,4,6,8] --holding_constant games_per_run_2 --constant_arena_size False --num_seeds_for_team 30 --generator_seed 1

python3 experiment_generator_rwg.py --experiment_directory /Users/mostafa/Documents/Code/PhD/TS-Platform/results/2021_10_06_b_aamas_data/2021_10_09_rwg_ctde_improved  --core_parameter_filename /Users/mostafa/Documents/Code/PhD/TS-Platform/src/default_slope_rwg_analysis_parameters.json --learning_type centralised --reward_level team --list_file_name LIST_CTDE_rwg --num_agents_in_setup [2] --num_seeds_for_team 1 --generator_seed 1 --many_architectures True
#python3 experiment_generator_rwg.py --experiment_directory /Users/mostafa/Documents/Code/PhD/TS-Platform/results/2021_10_06_b_aamas_data/2021_10_09_rwg_ctde_improved --core_parameter_filename /Users/mostafa/Documents/Code/PhD/TS-Platform/src/default_slope_rwg_analysis_parameters.json --learning_type fully-centralised --reward_level team --list_file_name LIST_FC_rwg --num_agents_in_setup [2] --num_seeds_for_team 1 --generator_seed 1 --many_architectures True
#python3 experiment_generator_rwg.py --experiment_directory /Users/mostafa/Documents/Code/PhD/TS-Platform/results/2021_10_06_b_aamas_data/dummy_rwg --core_parameter_filename /Users/mostafa/Documents/Code/PhD/TS-Platform/results/2021_10_06_b_aamas_data/dummy_rwg_params.json --learning_type centralised --reward_level team --list_file_name LIST_CTDE_rwg --num_agents_in_setup [2] --num_seeds_for_team 1 --generator_seed 1 --many_architectures True