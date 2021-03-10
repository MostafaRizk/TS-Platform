from experiment import run_experiment


experiment_list = "/Users/mostafa/Documents/Code/PhD/TS-Platform/results/2021_03_02_equal_games_per_learner/LIST_tester_local"
command_list = open(experiment_list, 'r').read().strip().split("\n")

parameters = []
for command in command_list:
    print(command)
    param = command.split(" ")[-1]
    print(param)
    parameters += [param]

run_experiment(parameters[0])