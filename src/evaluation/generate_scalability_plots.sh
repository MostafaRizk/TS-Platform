#!/bin/bash

for folder_name in "2021_10_06_b_aamas_data/2021_10_05_scalability_variable_arena_correct_neurons" #"2021_10_06_b_aamas_data/2021_10_04_scalability_constant_arena_correct_neurons" #"2021_10_06_b_aamas_data/2022_01_06_ctde_2_flat"
do
  episodes=30
  results_file="results_reevaluated_"$episodes"ep_final.csv"
  #python3 create_reevaluated_results.py --data_path /Users/mostafa/Documents/Code/PhD/TS-Platform/results/$folder_name/data --generation final --episodes $episodes --num_agents_to_remove 0 --min_team_size 2 --max_team_size 8 --env slope --bc_measure agent_action_count

  # Fitness plots
  #python3 plot_scalability.py --results_path /Users/mostafa/Documents/Code/PhD/TS-Platform/results/$folder_name/data/$results_file --graph_path /Users/mostafa/Documents/Code/PhD/TS-Platform/results/$folder_name/analysis/magic_plot_reeval_${episodes}_error_RCoop.pdf --plot_type error --max_agents 8 --violin False --y_height_min -10000 --y_height_max 20000 --showing fitness

  # Spec plots
  #python3 plot_scalability.py --results_path /Users/mostafa/Documents/Code/PhD/TS-Platform/results/$folder_name/data/$results_file --graph_path /Users/mostafa/Documents/Code/PhD/TS-Platform/results/$folder_name/analysis/magic_plot_reeval_${episodes}_spec_RCoop.pdf --plot_type error --max_agents 8 --violin False --showing specialisation

  # Rollout
  #python3 plot_episodes.py --results_path /Users/mostafa/Documents/Code/PhD/TS-Platform/results/$folder_name/data/$results_file --graph_path /Users/mostafa/Documents/Code/PhD/TS-Platform/results/$folder_name/analysis/episode_rollout_${episodes}_episodes_full_RCoop_left.pdf --min_agents 2 --max_agents 2 --y_height 30000 --exclude_middle_plots True
  #python3 plot_episodes.py --results_path /Users/mostafa/Documents/Code/PhD/TS-Platform/results/$folder_name/data/$results_file --graph_path /Users/mostafa/Documents/Code/PhD/TS-Platform/results/$folder_name/analysis/episode_rollout_${episodes}_episodes_full_RCoop_right.pdf --min_agents 8 --max_agents 8 --y_height 30000 --exclude_middle_plots True
  #python3 plot_episodes.py --results_path /Users/mostafa/Documents/Code/PhD/TS-Platform/results/$folder_name/data/$results_file --graph_path /Users/mostafa/Documents/Code/PhD/TS-Platform/results/$folder_name/analysis/episode_rollout_${episodes}_episodes_full_RCoop.pdf --min_agents 2 --max_agents 8 --y_height 30000 --exclude_middle_plots True

  # Histogram
  #python3 plot_coop_distribution.py --results_path /Users/mostafa/Documents/Code/PhD/TS-Platform/results/$folder_name/data/$results_file --graph_path /Users/mostafa/Documents/Code/PhD/TS-Platform/results/$folder_name/analysis/histogram_${episodes}_episodes_2_agents_averaged.pdf --agents 2 --averaged True
  #python3 plot_coop_distribution.py --results_path /Users/mostafa/Documents/Code/PhD/TS-Platform/results/$folder_name/data/$results_file --graph_path /Users/mostafa/Documents/Code/PhD/TS-Platform/results/$folder_name/analysis/histogram_${episodes}_episodes_8_agents_averaged.pdf --agents 8 --averaged True

  #python3 plot_scalability.py --results_path /Users/mostafa/Documents/Code/PhD/TS-Platform/results/$folder_name/data/$results_file --graph_path /Users/mostafa/Documents/Code/PhD/TS-Platform/results/$folder_name/analysis/magic_plot_reeval_${episodes}_best.png --plot_type best --max_agents 8 --violin False --y_height_min -10000 --y_height_max 40000 --showing fitness
  #python3 plot_scalability.py --results_path /Users/mostafa/Documents/Code/PhD/TS-Platform/results/$folder_name/data/$results_file --graph_path /Users/mostafa/Documents/Code/PhD/TS-Platform/results/$folder_name/analysis/magic_plot_reeval_${episodes}_participation.png --plot_type error --max_agents 8 --violin False --showing participation
  #python3 plot_scalability.py --results_path /Users/mostafa/Documents/Code/PhD/TS-Platform/results/$folder_name/data/$results_file --graph_path /Users/mostafa/Documents/Code/PhD/TS-Platform/results/$folder_name/analysis/magic_plot_reeval_${episodes}_violin_participation.png --plot_type None --max_agents 8 --violin True --showing participation

done
