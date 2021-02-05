from fitness import FitnessCalculator
from agents.nn_agent_lean import NNAgent
from array2gif import write_gif


def get_video_from_model(model_name, parameter_filename, save_directory, filename, reward_level="team", num_agents=2):
    fitness_calculator = FitnessCalculator(parameter_filename)
    results = None

    if reward_level == "team":
        full_genome = NNAgent.load_model_from_file(model_name)
        mid = int(len(full_genome) / num_agents)
        agent_list = []

        for i in range(num_agents):
            genome = full_genome[mid*i : mid*(i+1)]
            agent = NNAgent(fitness_calculator.get_observation_size(), fitness_calculator.get_action_size(),
                          parameter_filename, genome)
            agent_list += [agent]

        results = fitness_calculator.calculate_fitness(agent_list=agent_list, render=True, time_delay=0, render_mode="rgb_array")

    elif reward_level == "individual":
        agent_list = []

        for i in range(num_agents):
            genome = NNAgent.load_model_from_file(model_name)
            agent = NNAgent(fitness_calculator.get_observation_size(), fitness_calculator.get_action_size(),
                          parameter_filename, genome)
            agent_list += [agent]

        results = fitness_calculator.calculate_fitness(agent_list=agent_list, render=True, time_delay=0, render_mode="rgb_array")

    video_data = results['video_frames']
    save_to = f"{save_directory}/{filename}"
    write_gif(video_data, save_to, fps=5)

save_directory = "/Users/mostafa"

# FFNN getting stuck
'''
get_video_from_model(model_name="/Users/mostafa/Documents/Code/PhD/Reports/mid-candidature-2-report/slides/videos/ffnn/rwg_heterogeneous_team_nn_slope_1_2_4_1_4_8_4_1_3_7_1_3.0_0.2_2_1000_500_20_ffnn_False_2_4_tanh_20000_normal_0_1_-850.399999999998.npy",
                      parameter_filename="/Users/mostafa/Documents/Code/PhD/Reports/mid-candidature-2-report/slides/videos/ffnn/ffnn_no-bias_2HL_4HU_tanh.json",
                     save_directory=save_directory,
                     filename="ffnn.gif")
'''

# RNN succeeding
'''
get_video_from_model(model_name="/Users/mostafa/Documents/Code/PhD/Reports/mid-candidature-2-report/slides/videos/rnn/rwg_heterogeneous_team_nn_slope_1_2_4_1_4_8_4_1_3_7_1_3.0_0.2_2_1000_500_20_rnn_False_2_4_tanh_20000_normal_0_1_32133.419999999973.npy",
                      parameter_filename="/Users/mostafa/Documents/Code/PhD/Reports/mid-candidature-2-report/slides/videos/rnn/rnn_no-bias_2HL_4HU_tanh.json",
                     save_directory=save_directory,
                     filename="rnn.gif")
'''

# Centralised 2-agents best
'''
get_video_from_model(model_name="/Users/mostafa/Documents/Code/PhD/Reports/mid-candidature-2-report/slides/videos/centralised_slope_2_best/cma_heterogeneous_team_nn_slope_4005303369_2_4_1_4_8_4_1_3_7_1_3.0_0.2_2_1000_500_5_rnn_False_1_4_tanh_100_0.2_1000_0.001_200.0_95763.55999999987_final.npy",
                      parameter_filename="/Users/mostafa/Documents/Code/PhD/Reports/mid-candidature-2-report/slides/videos/centralised_slope_2_best/cma_heterogeneous_team_nn_slope_4005303369_2_4_1_4_8_4_1_3_7_1_3.0_0.2_2_1000_500_5_rnn_False_1_4_tanh_100_0.2_1000_0.001_200.0.json",
                     save_directory=save_directory,
                     filename="centralised_slope_2_best.gif")
'''
# Centralised 2-agents average
'''
get_video_from_model(model_name="/Users/mostafa/Documents/Code/PhD/Reports/mid-candidature-2-report/slides/videos/centralised_slope_2_average/cma_heterogeneous_team_nn_slope_2876537341_2_4_1_4_8_4_1_3_7_1_3.0_0.2_2_1000_500_5_rnn_False_1_4_tanh_100_0.2_1000_0.001_200.0_32644.000000000022_final.npy",
                      parameter_filename="/Users/mostafa/Documents/Code/PhD/Reports/mid-candidature-2-report/slides/videos/centralised_slope_2_average/cma_heterogeneous_team_nn_slope_2876537341_2_4_1_4_8_4_1_3_7_1_3.0_0.2_2_1000_500_5_rnn_False_1_4_tanh_100_0.2_1000_0.001_200.0.json",
                     save_directory=save_directory,
                     filename="centralised_slope_2_average.gif")
'''

# Decentralised 2-agents average
'''
get_video_from_model(model_name="/Users/mostafa/Documents/Code/PhD/Reports/mid-candidature-2-report/slides/videos/decentralised_slope_2_average/cma_heterogeneous_individual_nn_slope_117628830_2_4_1_4_8_4_1_3_7_1_3.0_0.2_2_1000_500_5_rnn_False_1_4_tanh_100_0.2_1000_0.001_200.0_60378.0_final.npy",
                      parameter_filename="/Users/mostafa/Documents/Code/PhD/Reports/mid-candidature-2-report/slides/videos/decentralised_slope_2_average/cma_heterogeneous_individual_nn_slope_117628830_2_4_1_4_8_4_1_3_7_1_3.0_0.2_2_1000_500_5_rnn_False_1_4_tanh_100_0.2_1000_0.001_200.0.json",
                     save_directory=save_directory,
                     filename="decentralised_slope_2_average.gif",
                     reward_level="individual")
'''

# Centralised 10-agents average
'''
get_video_from_model(model_name="/Users/mostafa/Documents/Code/PhD/Reports/mid-candidature-2-report/slides/videos/centralised_10_average/cma_heterogeneous_team_nn_slope_1261063144_10_20_1_4_8_20_1_3_7_1_3.0_0.2_2_1000_500_5_rnn_False_1_4_tanh_100_0.2_1000_0.001_0.0_103193.76000000004_final.npy",
                      parameter_filename="/Users/mostafa/Documents/Code/PhD/Reports/mid-candidature-2-report/slides/videos/centralised_10_average/cma_heterogeneous_team_nn_slope_1261063144_10_20_1_4_8_20_1_3_7_1_3.0_0.2_2_1000_500_5_rnn_False_1_4_tanh_100_0.2_1000_0.001_0.0.json",
                     save_directory=save_directory,
                     filename="centralised_10_average.gif",
                     reward_level="team",
                     num_agents=10)
'''

# Bad specialist
get_video_from_model(model_name="/Users/mostafa/Documents/Code/PhD/TS-Platform/results/2020_11_24_magic_plot_combined_new_seed/results/cma_heterogeneous_team_nn_slope_3771485675_2_4_1_4_8_4_1_3_7_1_3.0_0.2_2_1000_500_5_rnn_False_1_4_tanh_100_0.2_1000_0.001_200.0_36983.00000000002_final.npy",
                      parameter_filename="/Users/mostafa/Documents/Code/PhD/TS-Platform/results/2020_11_24_magic_plot_combined_new_seed/experiments/cma_heterogeneous_team_nn_slope_3771485675_2_4_1_4_8_4_1_3_7_1_3.0_0.2_2_1000_500_5_rnn_False_1_4_tanh_100_0.2_1000_0.001_200.0.json",
                     save_directory=save_directory,
                     filename="bad_specialist.gif")