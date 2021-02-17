'''
Visualise smooth fitness teams that had high fitness but low specialisation
'''

import numpy as np

from fitness import FitnessCalculator
from agents.nn_agent_lean import NNAgent

# Load all genomes from csv
file_path = "2HL_no-bias_sorted.csv"
parameter_filename = "rnn_no-bias_2HL_4HU_tanh.json"
rwg_data = []
f = open(file_path, "r")
rwg_data += f.read().strip().split("\n")
N_episodes = 20
spec_score_keys = ["R_coop", "R_coop_eff", "R_spec", "R_coop x P", "R_coop_eff x P", "R_spec x P"]
rwg_indices = [i for i in range(100)]  # This assumes the genomes are already sorted by mean fitness
num_spec_scores = len(spec_score_keys)*N_episodes
num_scores = N_episodes + num_spec_scores

# Get the least specialised solution in the top 100
min_spec = 1
mean_fitness = None
genome_to_vis = None
fitness_list = None
spec_list = None

for index in rwg_indices:
    row = rwg_data[index]
    genome = [float(element) for element in row.split(",")[0:-num_scores]]
    episode_scores = [float(score) for score in row.split(",")[-num_scores:-num_spec_scores]]

    # Make a list of lists. For each specialisation metric, there is a list of scores for all episodes
    raw_spec_scores = row.split(",")[-num_spec_scores:]
    raw_spec_scores = [[float(raw_spec_scores[i+j]) for i in range(0, len(raw_spec_scores), len(spec_score_keys))] for j in range(len(spec_score_keys))]

    rspec_index = spec_score_keys.index("R_spec")
    avg_spec = np.mean(raw_spec_scores[rspec_index])

    if avg_spec < min_spec:
        min_spec = avg_spec
        genome_to_vis = genome
        fitness_list = episode_scores
        mean_fitness = np.mean(fitness_list)
        spec_list = raw_spec_scores[rspec_index]

        # Exit loop if 0 spec is found
        if avg_spec == 0:
            break

# Visualise
print(min_spec)
print(spec_list)
print(mean_fitness)
print(fitness_list)
full_genome = genome_to_vis
fitness_calculator = FitnessCalculator(parameter_filename)

mid = int(len(full_genome) / 2)
genome_part_1 = full_genome[0:mid]
genome_part_2 = full_genome[mid:]
agent_1 = NNAgent(fitness_calculator.get_observation_size(), fitness_calculator.get_action_size(),
                  parameter_filename, genome_part_1)
agent_2 = NNAgent(fitness_calculator.get_observation_size(), fitness_calculator.get_action_size(),
                  parameter_filename, genome_part_2)
results = fitness_calculator.calculate_fitness(agent_list=[agent_1, agent_2], render=True, time_delay=0.1,
                                               measure_specialisation=True, logging=False, logfilename=None,
                                               render_mode="human")
print(results["fitness_matrix"])
print(results["specialisation_list"])