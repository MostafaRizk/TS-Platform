#! /usr/bin/env python

# PonyGE2
# Copyright (c) 2017 Michael Fenton, James McDermott,
#                    David Fagan, Stefan Forstenlechner,
#                    and Erik Hemberg
# Hereby licensed under the GNU GPL v3.
""" Python GE implementation """

from utilities.algorithm.general import check_python_version

check_python_version()

from stats.stats import get_stats
from algorithm.parameters import params, set_params
import sys

#For logging best rules
from os import path, getcwd, pardir
from utilities.algorithm.command_line_parser import parse_cmd_args


def mane():
    """ Run program """

     # Run evolution
    individuals = params['SEARCH_LOOP']()

    # Print final review
    get_stats(individuals, end=True)

    #Save best evolved rules
    cmd_args, unknown = parse_cmd_args(sys.argv[1:])
    if 'PARAMETERS' in cmd_args:
        swarm_files = ['aggregation.txt', 'foraging.txt', 'task_specialisation.txt']
        if cmd_args['PARAMETERS'] in swarm_files:
            best = max(individuals)
            ruleFilename = path.join(getcwd(), pardir, pardir, 'argos-code/results/' + cmd_args['PARAMETERS'].split('.')[0] + '_' + str(params['GENERATIONS']) + '_' + str(params['POPULATION_SIZE']) + '_' + str(params['SLOPE']) + "_" + str(params['RANDOM_SEED']) + '_best_rules.txt')
            rulefile = open(ruleFilename, 'w')
            rulefile.write(best.phenotype)
            rulefile.close()


if __name__ == "__main__":
    set_params(sys.argv[1:])  # exclude the ponyge.py arg itself
    mane()
