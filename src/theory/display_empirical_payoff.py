import json
import numpy as np

from fitness import FitnessCalculator
from agents.hardcoded.hitchhiker import HardcodedHitchhikerAgent
from agents.hardcoded.generalist import HardcodedGeneralistAgent
from agents.hardcoded.dropper import HardcodedDropperAgent
from agents.hardcoded.collector import HardcodedCollectorAgent

sliding_speed = 4
num_agents = 3
arena_width = num_agents*2
num_resources = arena_width

parameter_path = "default_empirical_parameters.json"
parameter_dictionary = json.loads(open(parameter_path).read())
parameter_dictionary["environment"]["slope"]["sliding_speed"] = sliding_speed
parameter_dictionary["environment"]["slope"]["num_agents"] = num_agents
parameter_dictionary["environment"]["slope"]["arena_width"] = arena_width
parameter_dictionary["environment"]["slope"]["num_resources"] = num_resources

f = open("temp.json", "w")
dictionary_string = json.dumps(parameter_dictionary, indent=4)
f.write(dictionary_string)
f.close()

fitness_calculator = FitnessCalculator("temp.json")
available_agents = [HardcodedHitchhikerAgent, HardcodedGeneralistAgent, HardcodedDropperAgent, HardcodedCollectorAgent]

if num_agents == 2:
    for class1 in available_agents:
        row = []
        for class2 in available_agents:
            agent1 = class1()
            agent2 = class2()
            agent_list = [agent1, agent2]
            #if type(agent1) == HardcodedCollectorAgent or type(agent2) == HardcodedCollectorAgent:
            if type(agent1) == HardcodedGeneralistAgent and type(agent2) == HardcodedCollectorAgent:
            #if False:
                render = True
                time_delay = 0.1
            else:
                render = False
                time_delay = 0
            results = fitness_calculator.calculate_fitness(agent_list=agent_list, render=render, time_delay=time_delay,
                                                               measure_specialisation=True, logging=False, logfilename=None,
                                                               render_mode="human")
            agent_scores = [round(np.mean(scores)) for scores in results['fitness_matrix']]
            row += [agent_scores]

        print(row)

if num_agents == 3:
    for class1 in available_agents:
        print("----")
        print(class1)
        for class2 in available_agents:
            row = []
            for class3 in available_agents:
                agent1 = class1()
                agent2 = class2()
                agent3 = class3()
                agent_list = [agent1, agent2, agent3]
                #if type(agent1) == type(agent2) == type(agent3) == HardcodedGeneralistAgent:
                #if type(agent2) == type(agent3) == HardcodedCollectorAgent:
                if type(agent1) == HardcodedDropperAgent and type(agent2) == HardcodedCollectorAgent and type(agent3) == HardcodedCollectorAgent:
                #if False:
                    render = True
                    time_delay = 0.1
                else:
                    render = False
                    time_delay = 0
                results = fitness_calculator.calculate_fitness(agent_list=agent_list, render=render, time_delay=time_delay,
                                                               measure_specialisation=True, logging=False, logfilename=None,
                                                               render_mode="human")
                agent_scores = [round(np.mean(scores)) for scores in results['fitness_matrix']]
                row += [agent_scores]

            print(row)

if num_agents == 4:
    for class1 in available_agents:
        print("----")
        print(class1)
        for class2 in available_agents:
            print(class2)
            for class3 in available_agents:
                row = []
                for class4 in available_agents:
                    agent1 = class1()
                    agent2 = class2()
                    agent3 = class3()
                    agent4 = class4()
                    agent_list = [agent1, agent2, agent3, agent4]
                    results = fitness_calculator.calculate_fitness(agent_list=agent_list, render=False, time_delay=0,
                                                                   measure_specialisation=True, logging=False, logfilename=None,
                                                                   render_mode="human")
                    agent_scores = [round(np.mean(scores)) for scores in results['fitness_matrix']]
                    row += [agent_scores]

                print(row)