#!/bin/bash

folder_name="2021_08_06_a_scalability_variable_arena"
agents_to_remove=(1 2 3 4)

#for i in ${!agents_to_remove[@]}
#do
#python3 create_reevaluated_results.py --data_path /Users/mostafa/Documents/Code/PhD/TS-Platform/results/$folder_name/data --generation final --episodes 30 --num_agents_to_remove ${agents_to_remove[$i]} --min_team_size 8 --max_team_size 8 --env slope --bc_measure agent_action_count
#done

python3 plot_robustness.py --data_path /Users/mostafa/Documents/Code/PhD/TS-Platform/results/$folder_name/data/ --file_list [results_reevaluated_30ep_final.csv,robustness_1_removed.csv,robustness_2_removed.csv,robustness_3_removed.csv,robustness_4_removed.csv] --graph_path /Users/mostafa/Documents/Code/PhD/TS-Platform/results/$folder_name/analysis/robustness_fitness.png --plot_type error --min_agents 8 --max_agents 8 --agents_removed 4 --y_min_height 0 --y_height 8000 --showing fitness
python3 plot_robustness.py --data_path /Users/mostafa/Documents/Code/PhD/TS-Platform/results/$folder_name/data/ --file_list [results_reevaluated_30ep_final.csv,robustness_1_removed.csv,robustness_2_removed.csv,robustness_3_removed.csv,robustness_4_removed.csv] --graph_path /Users/mostafa/Documents/Code/PhD/TS-Platform/results/$folder_name/analysis/robustness_spec.png --plot_type error --min_agents 8 --max_agents 8 --agents_removed 4 --showing specialisation