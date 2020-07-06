from agents.nn_agent import NNAgent
from fitness import FitnessCalculator

parameter_filenames = ["old_fitness_old_activation.json",
                       "new_fitness_old_activation.json"]
model_names = ["rwg_homogeneous_team_nn_1_2_3_1_4_8_4_1_3_7_0_1.0_1.0_1_1_500_5_0_linear_100000_normal_0_1_14.8.npy",
               "rwg_homogeneous_team_nn_1_2_3_1_4_8_4_1_3_7_1_3.0_0.2_2_1000_500_5_0_linear_100000_normal_0_1_13656.359999999993.npy"]
i = 1
fitness_calculator = FitnessCalculator(parameter_filenames[i])
genome = NNAgent.load_model_from_file(model_names[i])
agent_1 = NNAgent(fitness_calculator.get_observation_size(), fitness_calculator.get_action_size(), parameter_filenames[i], genome)
agent_2 = NNAgent(fitness_calculator.get_observation_size(), fitness_calculator.get_action_size(), parameter_filenames[i], genome)
fitness_calculator.calculate_fitness(agent_1, agent_2, render=True, time_delay=0.1)


