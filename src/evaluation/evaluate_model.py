"""
Evaluate the learned/evolved agent models with or without visualisation
"""

import argparse
import os
import numpy as np
import json

from fitness import FitnessCalculator
from agents.nn_agent_lean import NNAgent
from operator import add
from glob import glob


def evaluate_model(model_path, episodes, rendering=None, time_delay=None):
    #def evaluate_model(model_path, rendering="False", time_delay=0):
    if rendering == "True":
        rendering = True
    else:
        rendering = False

    if time_delay:
        time_delay = float(time_delay)
    else:
        time_delay = 0

    data_directory = "/".join(model_path.split("/")[:-1])
    model_filename = model_path.split("/")[-1]

    learning_type = model_filename.split("_")[0]
    algorithm_selected = model_filename.split("_")[1]

    if algorithm_selected != "rwg":
        generation = model_filename.split("_")[-1].strip(".npy")

        if learning_type == "centralised":
            parameter_list = model_filename.split("_")[:-2]
            parameter_filename = "_".join(parameter_list) + ".json"
            agent_index = "None"
        elif learning_type == "decentralised":
            parameter_list = model_filename.split("_")[:-3]
            parameter_filename = "_".join(parameter_list) + ".json"
            agent_index = model_filename.split("_")[-3]
        else:
            raise RuntimeError("Learning type must be centralised or decentralised")

    else:
        parameter_list = model_filename.split("_")[:-1]
        parameter_filename = "_".join(parameter_list) + ".json"
        agent_index = "None"

    parameter_path = os.path.join(data_directory, parameter_filename)
    parameter_dictionary = json.loads(open(parameter_path).read())
    environment = parameter_dictionary["general"]["environment"]
    num_agents = parameter_dictionary["environment"][environment]["num_agents"]
    agent_list = []

    if episodes and parameter_dictionary["environment"][environment]["num_episodes"] != episodes:
        parameter_dictionary["environment"][environment]["num_episodes"] = episodes
        new_parameter_path = os.path.join(data_directory, "temp.json")
        f = open(new_parameter_path, "w")
        dictionary_string = json.dumps(parameter_dictionary, indent=4)
        f.write(dictionary_string)
        f.close()
        fitness_calculator = FitnessCalculator(new_parameter_path)

    else:
        fitness_calculator = FitnessCalculator(parameter_path)

    if parameter_dictionary["general"]["learning_type"] == "centralised" and \
            parameter_dictionary["general"]["reward_level"] == "team":

        full_genome = np.load(model_path)

        for i in range(num_agents):
            start = i * int(len(full_genome) / num_agents)
            end = (i+1) * int(len(full_genome) / num_agents)
            sub_genome = full_genome[start:end]
            agent = NNAgent(fitness_calculator.get_observation_size(), fitness_calculator.get_action_size(),
                          parameter_path, sub_genome)
            agent_list += [agent]

    elif parameter_dictionary["general"]["learning_type"] == "centralised" and \
            parameter_dictionary["general"]["reward_level"] == "individual":

        genome_population = [np.load(model_path) for i in range(num_agents)]

        for genome in genome_population:
            agent = NNAgent(fitness_calculator.get_observation_size(),
                               fitness_calculator.get_action_size(), parameter_path, genome)
            agent_list += [agent]

    elif parameter_dictionary["general"]["learning_type"] == "decentralised" and \
            parameter_dictionary["general"]["reward_level"] == "individual":

        # Find genomes of all teammates
        genome_prefix = "_".join(str(item) for item in parameter_list)
        genome_suffix = f"{generation}.npy"
        all_genome_files = glob(f"{data_directory}/{genome_prefix}*{genome_suffix}")

        assert len(all_genome_files) == num_agents, "Number of genome files does not match number of agents on the team"

        for genome_file in all_genome_files:
            genome = np.load(genome_file)
            agent = NNAgent(fitness_calculator.get_observation_size(),
                            fitness_calculator.get_action_size(), parameter_path, genome)
            agent_list += [agent]

    results = fitness_calculator.calculate_fitness(agent_list=agent_list, render=rendering, time_delay=time_delay,
                                                   measure_specialisation=True, logging=False, logfilename=None,
                                                   render_mode="human")

    team_fitness_list = [0] * len(results['fitness_matrix'][0])

    for j in range(num_agents):
        team_fitness_list = list(map(add, team_fitness_list, results['fitness_matrix'][j]))

    team_score = np.mean(team_fitness_list)
    agent_scores = [np.mean(scores) for scores in results['fitness_matrix']]
    metric_index = 2  # R_spec
    specialisation = np.mean([spec[metric_index] for spec in results['specialisation_list']])

    #print(results['fitness_matrix'])
    #print(agent_scores)
    #print(team_score)
    #print(specialisation)

    return agent_scores, specialisation


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate models')
    parser.add_argument('--model_path', action="store")
    parser.add_argument('--episodes', action="store")
    parser.add_argument('--rendering', action="store")
    parser.add_argument('--time_delay', action="store")
    model_path = parser.parse_args().model_path
    episodes = int(parser.parse_args().episodes)
    rendering = parser.parse_args().rendering
    time_delay = parser.parse_args().time_delay

    evaluate_model(model_path, episodes, rendering, time_delay)
    #evaluate_model(model_path, rendering, time_delay)