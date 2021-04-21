import numpy as np

from fitness import FitnessCalculator
from agents.hardcoded.hitchhiker import HardcodedHitchhikerAgent
from agents.hardcoded.generalist import HardcodedGeneralistAgent
from agents.hardcoded.dropper import HardcodedDropperAgent
from agents.hardcoded.collector import HardcodedCollectorAgent

parameter_path = "default_parameters.json"
fitness_calculator = FitnessCalculator(parameter_path)
available_agents = [HardcodedHitchhikerAgent, HardcodedGeneralistAgent, HardcodedDropperAgent, HardcodedCollectorAgent]

for class1 in available_agents:
    row = []
    for class2 in available_agents:
        agent1 = class1()
        agent2 = class2()
        agent_list = [agent1, agent2]
        results = fitness_calculator.calculate_fitness(agent_list=agent_list, render=False, time_delay=0,
                                                           measure_specialisation=True, logging=False, logfilename=None,
                                                           render_mode="human")
        agent_scores = [np.mean(scores) for scores in results['fitness_matrix']]
        row += [agent_scores]

    print(row)