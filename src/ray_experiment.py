import ray
import argparse
import os
from experiment import run_experiment


@ray.remote
def experiment_wrapper(parameter_file):
    run_experiment(parameter_file)


parser = argparse.ArgumentParser(description='Run parallel experiments')
parser.add_argument('--experiment_list', action="store", dest="experiment_list")
experiment_list = parser.parse_args().experiment_list

ray.init(address=os.environ["ip_head"])
command_list = open(experiment_list, 'r').read().split("\n")

parameters = []
for command in command_list:
    print(command)
    param = command.split(" ")[-1]
    print(param)
    parameters += [param]

ray.get([experiment_wrapper.remote(param) for param in parameters])


