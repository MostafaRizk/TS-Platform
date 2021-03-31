#!/bin/bash

#python3 create_reevaluated_results.py --data_path /Users/mostafa/Documents/Code/PhD/TS-Platform/results/2021_03_22_equal_games_per_learner_sequential_reproducible/data --generation final
#python3 plot_scalability.py --results_path /Users/mostafa/Documents/Code/PhD/TS-Platform/results/2021_03_22_equal_games_per_learner_sequential_reproducible/data/results_reevaluated_final.csv --graph_path /Users/mostafa/Documents/Code/PhD/TS-Platform/results/2021_03_22_equal_games_per_learner_sequential_reproducible/analysis/magic_plot_reevaluated_6_agents.png --plot_type error --max_agents 6 --showing fitness
#python3 plot_scalability.py --results_path /Users/mostafa/Documents/Code/PhD/TS-Platform/results/2021_03_22_equal_games_per_learner_sequential_reproducible/data/results_reevaluated_final.csv --graph_path /Users/mostafa/Documents/Code/PhD/TS-Platform/results/2021_03_22_equal_games_per_learner_sequential_reproducible/analysis/magic_plot_reevaluated_spec_6_agents.png --plot_type error --max_agents 6 --showing specialisation

python3 create_reevaluated_results.py --data_path /Users/mostafa/Documents/Code/PhD/TS-Platform/results/2021_03_22_equal_games_per_learner_sequential_reproducible/data --generation final --episodes 50
#python3 plot_scalability.py --results_path /Users/mostafa/Documents/Code/PhD/TS-Platform/results/2021_03_22_equal_games_per_learner_sequential_reproducible/data/results_reevaluated_final.csv --graph_path /Users/mostafa/Documents/Code/PhD/TS-Platform/results/2021_03_22_equal_games_per_learner_sequential_reproducible/analysis/magic_plot_distributions.png --plot_type error --max_agents 6 --showing fitness
#python3 plot_scalability.py --results_path /Users/mostafa/Documents/Code/PhD/TS-Platform/results/2021_03_22_equal_games_per_learner_sequential_reproducible/data/results_reevaluated_final.csv --graph_path /Users/mostafa/Documents/Code/PhD/TS-Platform/results/2021_03_22_equal_games_per_learner_sequential_reproducible/analysis/magic_plot_distributions_spec.png --plot_type error --max_agents 6 --showing specialisation