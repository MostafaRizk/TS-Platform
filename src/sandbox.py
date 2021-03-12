import json
import os

directory = "/Users/mostafa/Documents/Code/PhD/TS-Platform/results/2021_03_02_equal_games_per_learner"
file = "tester_1.json"
parameter_filename = os.path.join(directory, file)
list_file_name = "LIST_tester"
g = open(os.path.join(directory, list_file_name), "a")
holding_constant = "games_per_learner"

if holding_constant == "games_per_run":
    pop_size_for_team = {
        "centralised": {2: 24},
        "decentralised": {2: 12}
    }
elif holding_constant == "games_per_learner":
    pop_size_for_team = {
        "centralised": {2: 10},
        "decentralised": {2: 10}
    }

for i in range(2, 6):
    parameter_dictionary = json.loads(open(parameter_filename).read())
    parameter_dictionary["general"]["seed"] = i
    parameter_dictionary["algorithm"]["agent_population_size"] = pop_size_for_team["centralised"][2]
    filename = f"tester_{i}.json"
    filepath = os.path.join(directory, filename)
    f = open(filepath, "w")
    dictionary_string = json.dumps(parameter_dictionary, indent=4)
    f.write(dictionary_string)
    f.close()
    g.write(f"python3 experiment.py --parameters {filename}\n")

for i in range(6, 11):
    parameter_dictionary = json.loads(open(parameter_filename).read())
    parameter_dictionary["general"]["seed"] = i
    parameter_dictionary["general"]["reward_level"] = "individual"
    parameter_dictionary["algorithm"]["agent_population_size"] = pop_size_for_team["centralised"][2]
    filename = f"tester_{i}.json"
    filepath = os.path.join(directory, filename)
    f = open(filepath, "w")
    dictionary_string = json.dumps(parameter_dictionary, indent=4)
    f.write(dictionary_string)
    f.close()
    g.write(f"python3 experiment.py --parameters {filename}\n")

for i in range(11, 16):
    learning_type = "decentralised"
    parameter_dictionary = json.loads(open(parameter_filename).read())
    parameter_dictionary["general"]["seed"] = i
    parameter_dictionary["general"]["reward_level"] = "individual"
    parameter_dictionary["algorithm"]["agent_population_size"] = pop_size_for_team[learning_type][2]
    parameter_dictionary["general"]["learning_type"] = learning_type
    filename = f"tester_{i}.json"
    filepath = os.path.join(directory, filename)
    f = open(filepath, "w")
    dictionary_string = json.dumps(parameter_dictionary, indent=4)
    f.write(dictionary_string)
    f.close()
    g.write(f"python3 experiment.py --parameters {filename}\n")

g.close()
