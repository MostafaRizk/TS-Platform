#!/bin/bash

python3 experiment_generator.py --experiment_directory /Users/mostafa/Documents/Code/PhD/TS-Platform/results/2021_06_07_a_tmaze_cma_2-agents --core_parameter_filename /Users/mostafa/Documents/Code/PhD/TS-Platform/src/default_tmaze_parameters.json --learning_type centralised --reward_level team --list_file_name LIST_2_agents --num_agents_in_setup [2] --holding_constant games_per_learner --num_seeds_for_team 30 --generator_seed 1