"""
Evaluate the learned/evolved agent models with or without visualisation
"""

import argparse
import os
import numpy as np
import json

from fitness import FitnessCalculator
from agents.nn_agent_lean import NNAgent


def evaluate_model(model_path, rendering, time_delay=0):
    if rendering == "True":
        rendering = True
    else:
        rendering = False

    time_delay = int(time_delay)

    data_directory = "/".join(model_path.split("/")[:-1])
    model_filename = model_path.split("/")[-1]
    parameter_filename = "_".join(model_filename.split("_")[:-2]) + ".json"
    parameter_path = os.path.join(data_directory, parameter_filename)
    fitness_calculator = FitnessCalculator(parameter_path)
    parameter_dictionary = json.loads(open(parameter_path).read())

    if parameter_dictionary["general"]["learning_type"] == "centralised" and \
            parameter_dictionary["general"]["reward_level"] == "team":
        full_genome = np.load(model_path)
        environment = parameter_dictionary["general"]["environment"]
        num_agents = parameter_dictionary["environment"][environment]["num_agents"]
        #mid = int(len(full_genome) / num_agents)
        agent_list = [None] * num_agents

        for i in range(num_agents):
            start = i * int(len(full_genome) / num_agents)
            end = (i+1) * int(len(full_genome) / num_agents)
            sub_genome = full_genome[start:end]
            agent_list[i] = NNAgent(fitness_calculator.get_observation_size(), fitness_calculator.get_action_size(),
                          parameter_path, sub_genome)

        results = fitness_calculator.calculate_fitness(agent_list=agent_list, render=rendering, time_delay=time_delay,
                                                       measure_specialisation=True, logging=False, logfilename=None,
                                                       render_mode="human")

        team_score = np.mean([sum([results['fitness_matrix'][agent][i] for agent in range(num_agents)]) for i in
                              range(len(results['fitness_matrix']))])
        metric_index = 2  # R_spec
        specialisation = np.mean([spec[metric_index] for spec in results['specialisation_list']])
        print(f"Team score: {team_score}")
        print(f"Specialisation: {specialisation}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate models')
    parser.add_argument('--model_path', action="store", dest="model_path")
    parser.add_argument('--rendering', action="store", dest="rendering")
    parser.add_argument('--time_delay', action="store", dest="time_delay")
    model_path = parser.parse_args().model_path
    rendering = parser.parse_args().rendering
    time_delay = parser.parse_args().time_delay

    evaluate_model(model_path, rendering, time_delay)