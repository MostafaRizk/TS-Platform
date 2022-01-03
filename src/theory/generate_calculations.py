import json
import sys
import os
import argparse

import numpy as np

from glob import glob


path = "/Users/mostafa/Documents/Code/PhD/TS-Platform/results/2021_11_09_a_game_theory/"
list_file = os.path.join(path, "LIST_game_theory")
g = open(list_file, "w")

for num_agents in range(2, 22, 2):
    # Freq of cooperation (standard): 2-max
    g.write(f"python3 sandbox.py --parameters distribution_params_b_2_agents_alignment=0.9.json --function get_distribution_agents --args_to_pass team_size_vs_coop --num_agents {num_agents}\n")

    # Freq of cooperation (partial=False): 2-max
    g.write(f"python3 sandbox.py --parameters distribution_params_c_2_agents_full_alignment_no_partial.json --function get_distribution_agents --args_to_pass team_size_vs_coop_no_partial --num_agents {num_agents}\n")

    # Freq of cooperation (partial=False, novices=-1): 2-max
    g.write(f"python3 sandbox.py --parameters distribution_params_d_2_agents_full_alignment_no_partial_less_novices.json --function get_distribution_agents --args_to_pass team_size_vs_coop_no_partial_less_novices --num_agents {num_agents}\n")

    # slopes 0, 2, 8
    for slope in [0, 2, 8]:
        # Price of anarchy (fitness): 2-max,
        g.write(f"python3 sandbox.py --parameters price_of_anarchy_a_fitness.json --function calculate_price_of_anarchy --args_to_pass fitness --num_agents {num_agents} --slope {slope}\n")
        # Price of anarchy (payoff): 2-max
        g.write(f"python3 sandbox.py --parameters price_of_anarchy_b_payoff.json --function calculate_price_of_anarchy --args_to_pass payoff --num_agents {num_agents} --slope {slope}\n")
        # Price of anarchy (no partial): 2-max
        g.write(f"python3 sandbox.py --parameters price_of_anarchy_c_payoff_no_partial.json --function calculate_price_of_anarchy --args_to_pass no_partial --num_agents {num_agents} --slope {slope}\n")

    # Slopes 1,3,4,5,6,7. agents=2,4,max
    if num_agents in [2, 4, 8, 20]:
        for slope in [1, 3, 4, 5, 6, 7]:
            # Price of anarchy (fitness)
            g.write(f"python3 sandbox.py --parameters price_of_anarchy_a_fitness.json --function calculate_price_of_anarchy --args_to_pass fitness --num_agents {num_agents} --slope {slope}\n")
            # Price of anarchy (payoff)
            g.write(f"python3 sandbox.py --parameters price_of_anarchy_b_payoff.json --function calculate_price_of_anarchy --args_to_pass payoff --num_agents {num_agents} --slope {slope}\n")
            # Price of anarchy (no partial)
            g.write(f"python3 sandbox.py --parameters price_of_anarchy_c_payoff_no_partial.json --function calculate_price_of_anarchy --args_to_pass no_partial --num_agents {num_agents} --slope {slope}\n")


g.close()



