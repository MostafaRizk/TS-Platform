#!/bin/bash

for folder_name in "2021_10_06_b_aamas_data/2021_10_05_scalability_variable_arena_correct_neurons" #"2021_10_06_b_aamas_data/2021_10_04_scalability_constant_arena_correct_neurons" #"2021_10_06_b_aamas_data/2021_08_06_a_scalability_variable_arena"
do
  episodes=30
  results_file="results_reevaluated_"$episodes"ep_final.csv"
  #python3 create_reevaluated_results.py --data_path /Users/mostafa/Documents/Code/PhD/TS-Platform/results/$folder_name/data --generation final --episodes $episodes --num_agents_to_remove 0 --min_team_size 2 --max_team_size 8 --env slope --bc_measure agent_action_count

  # Fitness plots
  python3 plot_scalability.py --results_path /Users/mostafa/Documents/Code/PhD/TS-Platform/results/$folder_name/data/$results_file --graph_path /Users/mostafa/Documents/Code/PhD/TS-Platform/results/$folder_name/analysis/magic_plot_reeval_${episodes}_error.pdf --plot_type error --max_agents 8 --violin False --y_height_min -10000 --y_height_max 20000 --showing fitness
  #python3 plot_scalability.py --results_path /Users/mostafa/Documents/Code/PhD/TS-Platform/results/$folder_name/data/$results_file --graph_path /Users/mostafa/Documents/Code/PhD/TS-Platform/results/$folder_name/analysis/magic_plot_reeval_${episodes}_mean.pdf --plot_type mean --max_agents 8 --violin False --y_height_min -10000 --y_height_max 20000 --showing fitness
  #python3 plot_scalability.py --results_path /Users/mostafa/Documents/Code/PhD/TS-Platform/results/$folder_name/data/$results_file --graph_path /Users/mostafa/Documents/Code/PhD/TS-Platform/results/$folder_name/analysis/magic_plot_reeval_${episodes}_violin_fitness.pdf --plot_type None --max_agents 8 --violin True --y_height_min -10000 --y_height_max 40000 --showing fitness

  # Spec plots
  python3 plot_scalability.py --results_path /Users/mostafa/Documents/Code/PhD/TS-Platform/results/$folder_name/data/$results_file --graph_path /Users/mostafa/Documents/Code/PhD/TS-Platform/results/$folder_name/analysis/magic_plot_reeval_${episodes}_spec_RSpec.pdf --plot_type error --max_agents 8 --violin False --showing specialisation
  #python3 plot_scalability.py --results_path /Users/mostafa/Documents/Code/PhD/TS-Platform/results/$folder_name/data/$results_file --graph_path /Users/mostafa/Documents/Code/PhD/TS-Platform/results/$folder_name/analysis/magic_plot_reeval_${episodes}_violin_RSpec.pdf --plot_type None --max_agents 8 --violin True --showing specialisation

  # Rollout
  python3 plot_episodes.py --results_path /Users/mostafa/Documents/Code/PhD/TS-Platform/results/$folder_name/data/$results_file --graph_path /Users/mostafa/Documents/Code/PhD/TS-Platform/results/$folder_name/analysis/episode_rollout_${episodes}_episodes.pdf --max_agents 8 --y_height 40000


  #python3 plot_scalability.py --results_path /Users/mostafa/Documents/Code/PhD/TS-Platform/results/$folder_name/data/$results_file --graph_path /Users/mostafa/Documents/Code/PhD/TS-Platform/results/$folder_name/analysis/magic_plot_reeval_${episodes}_best.png --plot_type best --max_agents 8 --violin False --y_height_min -10000 --y_height_max 40000 --showing fitness
  #python3 plot_scalability.py --results_path /Users/mostafa/Documents/Code/PhD/TS-Platform/results/$folder_name/data/$results_file --graph_path /Users/mostafa/Documents/Code/PhD/TS-Platform/results/$folder_name/analysis/magic_plot_reeval_${episodes}_participation.png --plot_type error --max_agents 8 --violin False --showing participation
  #python3 plot_scalability.py --results_path /Users/mostafa/Documents/Code/PhD/TS-Platform/results/$folder_name/data/$results_file --graph_path /Users/mostafa/Documents/Code/PhD/TS-Platform/results/$folder_name/analysis/magic_plot_reeval_${episodes}_violin_participation.png --plot_type None --max_agents 8 --violin True --showing participation

done