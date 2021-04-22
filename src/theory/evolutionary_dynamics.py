"""
Modelling slope foraging as a game and plotting the evolutionary dynamics

Strategies:

    Novice- Does nothing. Represents agents that are still learning

    Generalist- Gets resources from the source and brings them back to the nest

    Dropper- Drops resources from the source onto the slope

    Collector- Collects resources from the cache and deposits them at the nest
"""

import matplotlib.pyplot as plt
import json
import argparse
import os
import numpy as np

from scipy.optimize import differential_evolution
from scipy.integrate import odeint
from functools import partial
from scipy.stats import dirichlet

strategies = ["Novice", "Generalist", "Dropper", "Collector"]
cost = {}
num_agents = 0
generalist_reward = None  # The reward (i.e. resources retrieved) for an individual generalist
specialist_reward = None  # The reward for a dropper and collector pair together


def sample_distributions(num_samples=1000):
    """
    Sample a bunch of initial distributions. Each sample is an array of floats summing to 1. Each float represents the
    proportion of the population that uses that particular strategy.

    @param num_samples: Number of distributions to sample
    @return: List of lists of distributions
    """
    num_strategies = len(strategies)
    sample_format = np.ones(num_strategies)
    return dirichlet.rvs(size=num_samples, alpha=sample_format)


def theta(s):
    """
    Step function used to help compute payoff in different slopes
    @param s: Sliding speed (i.e. slope)
    @return: 1 if the slope is non-zero. 0 if slope is 0 (i.e. arena is flat)
    """
    if s == 0:
        return 0
    elif s > 0:
        return 1


def generate_constants(parameter_dictionary, slope=None):
    """
    Use the given parameters to generate the rewards and costs of each strategy.

    @param parameter_dictionary: Dictionary containing experiment parameters
    @param slope: Integer indicating slope. If none, uses the default from the dictionary
    @return: None
    """

    global cost
    global num_agents
    global generalist_reward
    global specialist_reward

    if slope is None:
        slope = parameter_dictionary["default_slope"]

    # Novice always has the same cost, the cost of being idle every time step.
    cost["Novice"] = parameter_dictionary["novice_base_cost"]

    # Generalist should have a slightly higher base cost (because it carries resources around) and an additional cost
    # proportional to the slope.
    cost["Generalist"] = parameter_dictionary["generalist_base_cost"] + \
                         parameter_dictionary["generalist_cost_multiplier"]*slope

    # Droppers have the same cost as novices when there's no slope because they don't carry things while moving and the
    # cost of going to the source is the same as idling if there's no slope. When there is a slope they pay an
    # additional cost proportional to it.
    cost["Dropper"] = parameter_dictionary["novice_base_cost"] + \
                      parameter_dictionary["dropper_cost_multiplier"]*slope

    # Collectors have the same cost as novices when there's no slope because no resources make it to the nest so they
    # are basically idle. When there is a slope they pay a small additional cost for moving resources that are
    # successfully dropped (if any). This cost is the same regardless of the slope.
    cost["Collector"] = parameter_dictionary["novice_base_cost"] + \
                        parameter_dictionary["collector_cost_multiplier"]*theta(slope)

    num_agents = parameter_dictionary["num_agents"]
    generalist_reward = parameter_dictionary["generalist_reward"]
    specialist_reward = parameter_dictionary["specialist_reward"]

    return


def get_payoff(strategy, teammate_combo):
    """
    Calculate payoff of a strategy when its teammates are a particular combination of other strategies

    @param strategy: Name of the agent's strategy as a string
    @param teammate_combo: List of strings representing the other agents' strategies
    @return: The part of the team payoff belonging to that particular agent
    """

    team = [strategy] + teammate_combo
    num_generalists = team.count("Generalist")
    num_droppers = team.count("Dropper")
    num_collectors = team.count("Collector")
    num_spec_pairs = min(num_droppers, num_collectors)

    payoff_from_generalists = (num_generalists * generalist_reward) / num_agents

    if num_collectors != 0:
        payoff_from_specialists = (num_spec_pairs * specialist_reward) / num_agents * (1 / num_collectors)

    else:
        payoff_from_specialists = 0

    strategy_cost = 0

    if strategy == "Collector":
        if num_spec_pairs == 0:
            strategy_cost = cost["Novice"]
        else:
            strategy_cost = cost["Collector"]
    else:
        strategy_cost = cost[strategy]

    return payoff_from_generalists + payoff_from_specialists - strategy_cost


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot evolutionary dynamics')
    parser.add_argument('--parameters', action="store", dest="parameters")

    parameter_file = parser.parse_args().parameters

    parameter_dictionary = json.loads(open(parameter_file).read())

    generate_constants(parameter_dictionary)

    for strategy1 in strategies:
        print("---")
        print(strategy1)
        for strategy2 in strategies:
            row = []
            for strategy3 in strategies:
                agent1 = round(get_payoff(strategy1, [strategy2, strategy3]))
                agent2 = round(get_payoff(strategy2, [strategy1, strategy3]))
                agent3 = round(get_payoff(strategy3, [strategy1, strategy2]))
                row += [f"({agent1}, {agent2}, {agent3})"]

            print(row)
