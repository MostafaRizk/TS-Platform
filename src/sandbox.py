import json
from glob import glob

from agents.hardcoded.collector import HardcodedCollectorAgent
from agents.hardcoded.dropper import HardcodedDropperAgent
from agents.hardcoded.generalist import HardcodedGeneralistAgent
from agents.nn_agent import NNAgent
from fitness import FitnessCalculator
from learning.rwg import RWGLearner
from learning.cma import CMALearner
from scripts import landscape_analysis
from scripts import plotter
import numpy as np

"""
parameter_filename = "cma_homogeneous_team_nn_117628830_2_3_1_4_8_4_1_3_7_1_3.0_0.2_2_1000_500_5_0_linear_80_0.2_5000_0.001_200.0.json"
fitness_calculator = FitnessCalculator(parameter_filename)
model_name = "cma_homogeneous_team_nn_117628830_2_3_1_4_8_4_1_3_7_1_3.0_0.2_2_1000_500_5_0_linear_80_0.2_5000_0.001_200.0_88131.83999999985_final.npy"
genome = NNAgent.load_model_from_file(model_name)
agent_1 = NNAgent(fitness_calculator.get_observation_size(), fitness_calculator.get_action_size(), parameter_filename, genome)
agent_2 = NNAgent(fitness_calculator.get_observation_size(), fitness_calculator.get_action_size(), parameter_filename, genome)
fitness_calculator.calculate_fitness(agent_1, agent_2, render=True, time_delay=0.1)
"""

"""

json_files = glob(f'json_files/*.json')
model_files = glob(f'json_files/final_models/*.npy')
results_file = "results_comparison2.csv"
results_header = "team_type,seed,fitness,specialisation\n"
f = open(results_file, "w")
f.write(results_header)

for parameter_filename in json_files:
    fitness_calculator = FitnessCalculator(parameter_filename)
    parameter_dictionary = json.loads(open(parameter_filename).read())
    agent_1 = HardcodedGeneralistAgent()
    agent_2 = HardcodedGeneralistAgent()
    results = fitness_calculator.calculate_fitness(agent_1, agent_2)
    fitness_1 = np.mean(results["fitness_1_list"])
    fitness_2 = np.mean(results["fitness_2_list"])
    team_fitness = fitness_1+fitness_2
    specialisation = np.mean(results["specialisation_list"])
    seed = parameter_dictionary["general"]["seed"]
    f.write(f"generalist,{seed},{team_fitness},{specialisation}\n")

    fitness_calculator = FitnessCalculator(parameter_filename)
    parameter_dictionary = json.loads(open(parameter_filename).read())
    agent_1 = HardcodedDropperAgent()
    agent_2 = HardcodedCollectorAgent()
    results = fitness_calculator.calculate_fitness(agent_1, agent_2)
    fitness_1 = np.mean(results["fitness_1_list"])
    fitness_2 = np.mean(results["fitness_2_list"])
    team_fitness = fitness_1 + fitness_2
    specialisation = np.mean(results["specialisation_list"])
    seed = parameter_dictionary["general"]["seed"]
    f.write(f"specialist,{seed},{team_fitness},{specialisation}\n")
    
    for model_name in model_files:
    parameter_filename = "_".join(model_name.replace("final_models/", "").split("_")[0:-2]) + ".json"
    fitness_calculator = FitnessCalculator(parameter_filename)
    parameter_dictionary = json.loads(open(parameter_filename).read())
    genome = NNAgent.load_model_from_file(model_name)
    agent_1 = NNAgent(fitness_calculator.get_observation_size(), fitness_calculator.get_action_size(),parameter_filename, genome)
    agent_2 = NNAgent(fitness_calculator.get_observation_size(), fitness_calculator.get_action_size(),parameter_filename, genome)
    results = fitness_calculator.calculate_fitness(agent_1, agent_2)
    fitness_1 = np.mean(results["fitness_1_list"])
    fitness_2 = np.mean(results["fitness_2_list"])
    team_fitness = fitness_1 + fitness_2
    specialisation = np.mean(results["specialisation_list"])
    seed = parameter_dictionary["general"]["seed"]
    f.write(f"evolved,{seed},{team_fitness},{specialisation}\n")

f.close()
"""



# Create header
"""
for i in range(20,5020,20):
    results_file = f"results_files/results_{i}.csv"
    parameter_filename = "cma_homogeneous_team_nn_117628830_2_3_1_4_8_4_1_3_7_1_3.0_0.2_2_1000_500_5_0_linear_80_0.2_5000_0.001_200.0.json"
    parameter_dictionary = json.loads(open(parameter_filename).read())
    result_headings = CMALearner.get_results_headings(parameter_dictionary)
    result_headings += ["seed_fitness", "fitness", "model_name"]
    result_headings = ",".join(result_headings)
    result_headings += "\n"
    f = open(results_file, "r")
    file_contents = f.read()
    f.close()
    f = open(results_file, "w")
    f.write(result_headings)
    f.write(file_contents)
    f.close()
"""


#landscape_analysis.plot_fitness_distribution("results_final.csv", "fitness_distribution.png")
#plotter.plot_evolution_history("results_files", "evolution_history.png")

parameter_filename = "cma_homogeneous_team_nn_117628830_2_3_1_4_8_4_1_3_7_1_3.0_0.2_2_1000_500_5_0_linear_80_0.2_5000_0.001_200.0.json"
fitness_calculator = FitnessCalculator(parameter_filename)
agent_1 = HardcodedGeneralistAgent()
agent_2 = HardcodedGeneralistAgent()
results = fitness_calculator.calculate_fitness(agent_1, agent_2, render=True, time_delay=0.1)