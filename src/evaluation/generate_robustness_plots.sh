#!/bin/bash

for folder_name in "2021_10_06_b_aamas_data/2021_10_05_scalability_variable_arena_correct_neurons" #"2021_10_06_b_aamas_data/2021_08_06_a_scalability_variable_arena"
do
agents_to_remove=(1 2 3 4)

#for i in ${!agents_to_remove[@]}
#do
#python3 create_reevaluated_results.py --data_path /Users/mostafa/Documents/Code/PhD/TS-Platform/results/$folder_name/data --generation final --episodes 30 --num_agents_to_remove ${agents_to_remove[$i]} --dummy_observations True --min_team_size 8 --max_team_size 8 --env slope --bc_measure agent_action_count
#done

python3 plot_robustness.py --data_path /Users/mostafa/Documents/Code/PhD/TS-Platform/results/$folder_name/data/ --file_list [results_reevaluated_30ep_final.csv,robustness_1_removed_dummy=True.csv,robustness_2_removed_dummy=True.csv,robustness_3_removed_dummy=True.csv,robustness_4_removed_dummy=True.csv] --graph_path /Users/mostafa/Documents/Code/PhD/TS-Platform/results/$folder_name/analysis/robustness_fitness_dummy=True.pdf --plot_type error --min_agents 8 --max_agents 8 --agents_removed 4 --y_min_height -1000 --y_height 8000 --showing fitness
python3 plot_robustness.py --data_path /Users/mostafa/Documents/Code/PhD/TS-Platform/results/$folder_name/data/ --file_list [results_reevaluated_30ep_final.csv,robustness_1_removed_dummy=True.csv,robustness_2_removed_dummy=True.csv,robustness_3_removed_dummy=True.csv,robustness_4_removed_dummy=True.csv] --graph_path /Users/mostafa/Documents/Code/PhD/TS-Platform/results/$folder_name/analysis/robustness_spec_dummy=True.pdf --plot_type error --min_agents 8 --max_agents 8 --agents_removed 4 --y_min_height -1000 --y_height 8000 --showing specialisation

done