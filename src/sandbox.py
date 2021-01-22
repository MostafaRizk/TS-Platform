import json
import numpy as np

from fitness import FitnessCalculator
from agents.hardcoded.generalist import HardcodedGeneralistAgent
from agents.hardcoded.dropper import HardcodedDropperAgent
from agents.hardcoded.collector import HardcodedCollectorAgent
from agents.hardcoded.hitchhiker import HardcodedHitchhikerAgent

specialisation_list = []

for seed in range(1):
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
    agent_3 = HardcodedHitchhikerAgent()
    results = fitness_calculator.calculate_fitness(agent_list=[agent_1, agent_2, agent_3], render=True, time_delay=0.1, measure_specialisation=True, logging=False, logfilename=None, render_mode="human")
    specialisation_list += results["specialisation_list"]

print(np.mean(np.array(specialisation_list), axis=0))