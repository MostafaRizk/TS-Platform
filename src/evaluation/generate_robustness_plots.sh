#!/bin/bash

#python3 create_reevaluated_results.py --data_path /Users/mostafa/Documents/Code/PhD/TS-Platform/results/2021_04_09_b_magic_plot_combined/data --generation final --episodes 5 --num_agents_to_remove 1 --max_team_size 4
#python3 create_reevaluated_results.py --data_path /Users/mostafa/Documents/Code/PhD/TS-Platform/results/2021_04_09_b_magic_plot_combined/data --generation final --episodes 5 --num_agents_to_remove 2 --max_team_size 4
#python3 create_reevaluated_results.py --data_path /Users/mostafa/Documents/Code/PhD/TS-Platform/results/2021_04_09_b_magic_plot_combined/data --generation final --episodes 5 --num_agents_to_remove 3 --max_team_size 4
#python3 create_reevaluated_results.py --data_path /Users/mostafa/Documents/Code/PhD/TS-Platform/results/2021_04_09_b_magic_plot_combined/data --generation final --episodes 5 --num_agents_to_remove 0 --max_team_size 4


#python3 create_reevaluated_results.py --data_path /Users/mostafa/Documents/Code/PhD/TS-Platform/results/2021_04_20_homogeneous_evolution/data --generation final --episodes 5 --num_agents_to_remove 0 --max_team_size 4
#python3 create_reevaluated_results.py --data_path /Users/mostafa/Documents/Code/PhD/TS-Platform/results/2021_04_20_homogeneous_evolution/data --generation final --episodes 5 --num_agents_to_remove 1 --max_team_size 4
#python3 create_reevaluated_results.py --data_path /Users/mostafa/Documents/Code/PhD/TS-Platform/results/2021_04_20_homogeneous_evolution/data --generation final --episodes 5 --num_agents_to_remove 2 --max_team_size 4
#python3 create_reevaluated_results.py --data_path /Users/mostafa/Documents/Code/PhD/TS-Platform/results/2021_04_20_homogeneous_evolution/data --generation final --episodes 5 --num_agents_to_remove 3 --max_team_size 4

#python3 plot_robustness.py --data_path /Users/mostafa/Documents/Code/PhD/TS-Platform/results/2021_04_09_b_magic_plot_combined/data/ --file_list [robustness_0_removed.csv,robustness_1_removed.csv,robustness_2_removed.csv,robustness_3_removed.csv] --graph_path /Users/mostafa/Documents/Code/PhD/TS-Platform/results/2021_04_09_b_magic_plot_combined/analysis/robustness_of_fitness.png --plot_type error --max_agents 4 --y_height 8000 --showing fitness
#python3 plot_robustness.py --data_path /Users/mostafa/Documents/Code/PhD/TS-Platform/results/2021_04_09_b_magic_plot_combined/data/ --file_list [robustness_0_removed.csv,robustness_1_removed.csv,robustness_2_removed.csv,robustness_3_removed.csv] --graph_path /Users/mostafa/Documents/Code/PhD/TS-Platform/results/2021_04_09_b_magic_plot_combined/analysis/robustness_of_specialisation_participation.png --plot_type error --max_agents 4 --showing specialisation

python3 plot_robustness.py --data_path /Users/mostafa/Documents/Code/PhD/TS-Platform/results/2021_04_29_magic_combined_homogeneous/data/ --file_list [robustness_0_removed.csv,robustness_1_removed.csv,robustness_2_removed.csv,robustness_3_removed.csv] --graph_path /Users/mostafa/Documents/Code/PhD/TS-Platform/results/2021_04_29_magic_combined_homogeneous/analysis/robustness_of_fitness.png --plot_type error --max_agents 4 --y_height 8000 --showing fitness
#python3 plot_robustness.py --data_path /Users/mostafa/Documents/Code/PhD/TS-Platform/results/2021_04_29_magic_combined_homogeneous/data/ --file_list [robustness_0_removed.csv,robustness_1_removed.csv,robustness_2_removed.csv,robustness_3_removed.csv] --graph_path /Users/mostafa/Documents/Code/PhD/TS-Platform/results/2021_04_29_magic_combined_homogeneous/analysis/robustness_of_specialisation_participation.png --plot_type error --max_agents 4 --showing specialisation
