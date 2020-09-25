import numpy as np
import json

from learning.learner_parent import Learner
from learning.rwg import RWGLearner


def get_best_of_k_samples(results_file, parameter_filename, k, N_episodes):
    """
    From a results file with n samples. Find the best among the first k samples and write it to a file

    @param results_file: CSV file containing n models
    @param parameter_filename: Parameter file used to generate results
    @param k: An integer
    @param N_episodes: Number of episodes used to evaluate the sample
    """

    # Load json and csv files
    parameter_dictionary = json.loads(open(parameter_filename).read())
    f = open(results_file, "r")
    data = f.read().strip().split("\n")

    # Get best genome in first k samples
    best_genome = np.array([float(element) for element in data[0].split(",")[0:-N_episodes]])
    best_score = np.mean([float(episode_score) for episode_score in data[0].split(",")[-N_episodes:]])

    for row in data[0:k]:
        genome = np.array([float(element) for element in row.split(",")[0:-N_episodes]])
        episode_scores = [float(score) for score in row.split(",")[-N_episodes:]]
        mean_score = np.mean(episode_scores)

        if mean_score > best_score:
            best_score = mean_score
            best_genome = genome

    # Generate model name
    if parameter_dictionary['general']['reward_level'] == "team":
        parameter_dictionary['algorithm']['agent_population_size'] = k * 2
    else:
        parameter_dictionary['algorithm']['agent_population_size'] = k

    parameters_in_name = Learner.get_core_params_in_model_name(parameter_dictionary)
    parameters_in_name += RWGLearner.get_additional_params_in_model_name(parameter_dictionary)
    parameters_in_name += [best_score]
    model_name = "_".join([str(param) for param in parameters_in_name])

    np.save(model_name, best_genome)


get_best_of_k_samples(
    "all_genomes_rwg_heterogeneous_team_nn_slope_1_2_4_1_4_8_4_1_3_7_1_3.0_0.2_2_1000_500_20_rnn_False_1_4_tanh_20000_normal_0_1_.csv",
    "rnn_no-bias_1HL_4HU_tanh.json", k=1000, N_episodes=20)
