#!/bin/bash

#python3 create_reevaluated_results.py --data_path /Users/mostafa/Documents/Code/PhD/TS-Platform/results/2021_04_09_b_magic_plot_combined/data --generation final --episodes 50
#python3 plot_scalability.py --results_path /Users/mostafa/Documents/Code/PhD/TS-Platform/results/2021_04_09_b_magic_plot_combined/data/results_reevaluated_final.csv --graph_path /Users/mostafa/Documents/Code/PhD/TS-Platform/results/2021_04_09_b_magic_plot_combined/analysis/magic_plot_reevaluated_8_agents_50_episodes.png --plot_type error --max_agents 8 --violin False --y_height 8000 --showing fitness
#python3 plot_scalability.py --results_path /Users/mostafa/Documents/Code/PhD/TS-Platform/results/2021_04_09_b_magic_plot_combined/data/results_reevaluated_final.csv --graph_path /Users/mostafa/Documents/Code/PhD/TS-Platform/results/2021_04_09_b_magic_plot_combined/analysis/magic_plot_reevaluated_spec_8_agents__50_episodes.png --plot_type error --max_agents 8 --violin False --showing specialisation
#python3 plot_scalability.py --results_path /Users/mostafa/Documents/Code/PhD/TS-Platform/results/2021_04_09_b_magic_plot_combined/data/results_reevaluated_final.csv --graph_path /Users/mostafa/Documents/Code/PhD/TS-Platform/results/2021_04_09_b_magic_plot_combined/analysis/magic_plot_reevaluated_8_agents_50_episodes_violin.png --plot_type error --max_agents 8 --violin True --y_height 8000 --showing fitness

#python3 plot_scalability.py --results_path /Users/mostafa/Documents/Code/PhD/TS-Platform/results/2021_04_09_b_magic_plot_combined/data/results_reevaluated_final.csv --graph_path /Users/mostafa/Documents/Code/PhD/TS-Platform/results/2021_04_09_b_magic_plot_combined/analysis/magic_plot_reevaluated_8_agents_50_episodes_no_participation.png --plot_type error --max_agents 8 --violin False --y_height 8000 --showing fitness
#python3 plot_scalability.py --results_path /Users/mostafa/Documents/Code/PhD/TS-Platform/results/2021_04_09_b_magic_plot_combined/data/results_reevaluated_final.csv --graph_path /Users/mostafa/Documents/Code/PhD/TS-Platform/results/2021_04_09_b_magic_plot_combined/analysis/magic_plot_reevaluated_8_agents_50_episodes_no_participation_spec.png --plot_type error --max_agents 8 --violin False --showing specialisation
#python3 plot_episodes.py --results_path /Users/mostafa/Documents/Code/PhD/TS-Platform/results/2021_04_09_b_magic_plot_combined/data/results_reevaluated_final.csv --graph_path /Users/mostafa/Documents/Code/PhD/TS-Platform/results/2021_04_09_b_magic_plot_combined/analysis/episode_rollout_50_episodes.png --max_agents 8 --y_height 20000

#python3 create_reevaluated_results.py --data_path /Users/mostafa/Documents/Code/PhD/TS-Platform/results/2021_04_20_homogeneous_evolution/data --generation final --episodes 50 --num_agents_to_remove 0 --max_team_size 8