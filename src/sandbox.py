import argparse
import json

from agents.hardcoded.collector import HardcodedCollectorAgent
from agents.hardcoded.dropper import HardcodedDropperAgent
from agents.hardcoded.generalist import HardcodedGeneralistAgent
from agents.hardcoded.lazy_generalist import HardcodedLazyGeneralistAgent
from fitness import FitnessCalculator
from learning.cma import CMALearner
from learning.rwg import RWGLearner
import numpy as np
from agents.nn_agent_lean import NNAgent
import pandas as pd

scores = []
for seed in range(30):
    parameter_filename = "default_parameters.json"
    parameter_dictionary = json.loads(open(parameter_filename).read())
    parameter_dictionary['general']['seed'] = seed
    f = open(parameter_filename, "w")
    dictionary_string = json.dumps(parameter_dictionary, indent=4)
    f.write(dictionary_string)
    f.close()
    fitness_calculator = FitnessCalculator(parameter_filename)

    agent_1 = HardcodedDropperAgent()
    agent_2 = HardcodedCollectorAgent()
    results = fitness_calculator.calculate_fitness(agent_1, agent_2, render=False)
    fitness_1_list = results["fitness_1_list"]
    fitness_2_list = results["fitness_2_list"]
    specialisation_list = results["specialisation_list"]
    zipped_list = zip(fitness_1_list, fitness_2_list)
    scores += [np.mean([fitness_1 + fitness_2 for (fitness_1, fitness_2) in zipped_list])]

print(f"{np.mean(scores)}")