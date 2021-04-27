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
import pandas as pd
import numpy as np
import math
import cma
import time
import bisect

from scipy.optimize import differential_evolution
from scipy.optimize import minimize
from scipy.optimize import Bounds
from scipy.optimize import LinearConstraint
from scipy.integrate import odeint
from functools import partial
from scipy.stats import dirichlet
from itertools import combinations_with_replacement

strategies = ["Novice", "Generalist", "Dropper", "Collector"]
strategy_id = {"Novice": 0, "Generalist": 1, "Dropper": 2, "Collector": 3}
cost = {}
payoff_dict = {}
num_agents = 0
generalist_reward = None  # The reward (i.e. resources retrieved) for an individual generalist
specialist_reward = None  # The reward for a dropper and collector pair together
combos = None  # Sorted list of possible strategy combinations for num_agents-1 agents
full_combos = {}  # For each strategy s, combo list from before but with s inserted
slope_list = [i for i in range(9)]
team_list = [i for i in range(2, 9)]
random_state = None
distribution_precision = 4  # Number of decimal places when expressing the proportion of each strategy in the population
#archive = {}

prices_of_anarchy = {}

for slope in slope_list:

    temp_dict = {}

    for agent in team_list:
        temp_dict[agent] = None

    prices_of_anarchy[slope] = temp_dict


def sample_distributions(num_samples=1000):
    """
    Sample a bunch of initial distributions. Each sample is an array of floats summing to 1. Each float represents the
    proportion of the population that uses that particular strategy.

    @param num_samples: Number of distributions to sample
    @return: List of lists of distributions
    """
    num_strategies = len(strategies)
    sample_format = np.ones(num_strategies)
    return dirichlet.rvs(size=num_samples, alpha=sample_format, random_state=random_state)


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


def generate_constants(parameter_dictionary, team_size=None, slope=None):
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
    global combos
    global full_combos
    global random_state

    if slope is None:
        slope = parameter_dictionary["default_slope"]

    # Novice always has the same cost, the cost of being idle every time step.
    cost["Novice"] = parameter_dictionary["novice_base_cost"]

    # Generalist should have a slightly higher base cost (because it carries resources around) and an additional cost
    # proportional to the slope.
    cost["Generalist"] = parameter_dictionary["generalist_base_cost"] + \
                         parameter_dictionary["generalist_cost_multiplier"] * slope

    # Droppers have the same cost as novices when there's no slope because they don't carry things while moving and the
    # cost of going to the source is the same as idling if there's no slope. When there is a slope they pay an
    # additional cost proportional to it.
    cost["Dropper"] = parameter_dictionary["novice_base_cost"] + \
                      parameter_dictionary["dropper_cost_multiplier"] * slope

    # Collectors have the same cost as novices when there's no slope because no resources make it to the nest so they
    # are basically idle. When there is a slope they pay a small additional cost for moving resources that are
    # successfully dropped (if any). This cost is the same regardless of the slope.
    cost["Collector"] = parameter_dictionary["novice_base_cost"] + \
                        parameter_dictionary["collector_cost_multiplier"] * theta(slope)

    if team_size is None:
        num_agents = parameter_dictionary["default_num_agents"]
    else:
        num_agents = team_size

    generalist_reward = parameter_dictionary["generalist_reward"]
    specialist_reward = parameter_dictionary["specialist_reward"] * theta(slope)

    sorted_strategies = strategies[:]
    sorted_strategies.sort()

    combos = list(combinations_with_replacement(sorted_strategies, num_agents-1))

    '''for strat in strategies:
        expanded_combos = [None]*len(combos)

        for i,combo in enumerate(combos):
            new_combo = list(combo[:])
            bisect.insort(new_combo, strat)
            expanded_combos[i] = tuple(new_combo)

        full_combos[strat] = expanded_combos'''


    random_state = parameter_dictionary["random_state"]

    return


def get_payoff(team, all=False):
    """
    Calculate payoff of every strategy on a team or just first agent

    @param team: List of strings representing the agents' strategies
    @param all: Boolean denoting if returning all payoffs
    @return: The summed payoff (reward - cost) for each agent on the team or payoff for just the first agent
    """

    #key = str(team)

    #if key in payoff_dict:
    #    return payoff_dict[key]

    num_generalists = team.count("Generalist")
    num_droppers = team.count("Dropper")
    num_collectors = team.count("Collector")
    num_spec_pairs = min(num_droppers, num_collectors)

    payoff_from_generalists = (num_generalists * generalist_reward) / num_agents

    if num_collectors != 0:
        payoff_from_specialists = (num_spec_pairs * specialist_reward) / num_agents

    else:
        payoff_from_specialists = 0

    rewards = payoff_from_generalists + payoff_from_specialists

    payoff = 0

    for strategy in team:
        if strategy == "Collector":
            if num_spec_pairs == 0:
                strategy_cost = cost["Novice"]
            else:
                strategy_cost = cost["Collector"]
        else:
            strategy_cost = cost[strategy]

        if all:
            payoff += rewards - strategy_cost

        else:
            payoff = rewards - strategy_cost
            break

    #payoff_dict[key] = payoff
    return payoff


def inspect_payoff_function(parameter_dictionary):
    """
    Visualise payoff matrices (assumes parameter dictionary sets 3 agents)

    @param parameter_dictionary:
    @return: None
    """
    generate_constants(parameter_dictionary)

    for strategy1 in strategies:
        print("----------------------------------------------------------------")
        print(strategy1)
        matrix = []

        for strategy2 in strategies:
            row = []

            for strategy3 in strategies:
                agent1 = round(get_payoff(strategy1, [strategy2, strategy3]))
                agent2 = round(get_payoff(strategy2, [strategy1, strategy3]))
                agent3 = round(get_payoff(strategy3, [strategy1, strategy2]))
                row += [f"({agent1}, {agent2}, {agent3})"]

            matrix += [row]

        df = pd.DataFrame(matrix, strategies, strategies)
        df.style.set_properties(**{'text-align': 'center'})

        with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', None):
            print(df)


def multinomial_coefficient(combo):
    """
    Given a list of agent strategies, calculate the multionomial coefficient i.e. the number of ways those strategies
    can be ordered.

    e.g.
    multinomial_coefficient(["Dropper", "Collector"]) returns 2 because you can have:

    Dropper, Collector
    Collector, Dropper

    multinomial_coefficient(["Dropper", "Generalist", "Collector"]) returns 6

    multinomial_coefficient(["Dropper", "Dropper"]) returns 1

    @param combo: List of strategies of a group of agents
    @return:
    """
    n = len(combo)
    denominator = 1
    unique_strategies = {}

    for agent_strategy in combo:
        if agent_strategy in unique_strategies:
            unique_strategies[agent_strategy] += 1
        else:
            unique_strategies[agent_strategy] = 1

    for key in unique_strategies:
        denominator *= math.factorial(unique_strategies[key])

    # TODO: Can this be more efficient?
    return math.factorial(n) // denominator


def get_probability_of_teammate_combo(teammate_combo, probability_of_strategies):
    """
    Calculate the probability of a particular set of teammates

    @param teammate_combo: List of teammate strategies
    @param probability_of_strategies: List of probabilities of sampling each strategy from the infinite population
    @return: Float representing probability
    """
    product = 1.0

    for strategy in teammate_combo:
        product *= probability_of_strategies[strategy_id[strategy]]

    return product * multinomial_coefficient(teammate_combo)


def get_fitnesses(P):
    combo_probabilities = {}

    for combo in combos:
        combo_probabilities[combo] = get_probability_of_teammate_combo(combo, P)


    f_novice = sum([get_payoff(["Novice"] + list(combo)) * combo_probabilities[combo] for combo in combos])
    f_generalist = sum([get_payoff(["Generalist"] + list(combo)) * combo_probabilities[combo] for combo in combos])
    f_dropper = sum([get_payoff(["Dropper"] + list(combo)) * combo_probabilities[combo] for combo in combos])
    f_collector = sum([get_payoff(["Collector"] + list(combo)) * combo_probabilities[combo] for combo in combos])
    f_avg = P[0] * f_novice + P[1] * f_generalist + P[2] * f_dropper + P[3] * f_collector
    '''
    f_novice = sum([get_payoff(full_combos["Novice"][index]) * combo_probabilities[combo] for index,combo in enumerate(combos)])
    f_generalist = sum([get_payoff(full_combos["Generalist"][index]) * combo_probabilities[combo] for index,combo in enumerate(combos)])
    f_dropper = sum([get_payoff(full_combos["Dropper"][index]) * combo_probabilities[combo] for index,combo in enumerate(combos)])
    f_collector = sum([get_payoff(full_combos["Collector"][index]) * combo_probabilities[combo] for index,combo in enumerate(combos)])
    f_avg = P[0] * f_novice + P[1] * f_generalist + P[2] * f_dropper + P[3] * f_collector'''

    return f_novice, f_generalist, f_dropper, f_collector, f_avg


def get_change(P, t=0):
    """
    Function to be passed to ODEINT. Calculates replicator dynamic equation for all strategies.
    i.e. Calculates rate of change for each strategy
    e.g.

    Where

    x_dropper = the percentage of the population that are droppers,

    x_dropper' = the rate of increase/decrease of the number of dropper strategies in the population

    f(x_dropper) = the fitness of a dropper in this population (depends on the distribution of other strategies),
    calculated by getting the payoff of a dropper when it is paired with each other strategy then taking the weighted
    sum, where the weight is the probability of being paired with each strategy

    avg_fitness = the average fitness of all other strategies

    We then get

    x_dropper' = x_dropper(f(x_dropper) - avg_fitness)

    @param P: The distribution of novice, generalist, dropper and collector strategies (in that order) at this time step
    @param t: The current time step
    @return: The derivative of each strategy
    """

    f_novice, f_generalist, f_dropper, f_collector, f_avg = get_fitnesses(P)

    return np.array([
        P[0] * (f_novice - f_avg),
        P[1] * (f_generalist - f_avg),
        P[2] * (f_dropper - f_avg),
        P[3] * (f_collector - f_avg)
    ])


def get_many_final_distributions(parameter_dictionary):
    """
    Samples several initial population distributions and calculates the final distribution of each using the replicator
    dynamic.

    @param parameter_dictionary:
    @return: List of lists where each list is a final population distribution
    """
    total_time = parameter_dictionary["total_time"]
    intervals = parameter_dictionary["intervals"]
    num_samples = parameter_dictionary["num_samples"]

    t = np.linspace(0, intervals, total_time)
    initial_distributions = sample_distributions(num_samples)
    final_distributions = [None] * num_samples

    for index in range(len(initial_distributions)):
        P0 = initial_distributions[index]
        dist = odeint(get_change, P0, t)[-1]

        for i in range(len(dist)):
            dist[i] = round(dist[i], distribution_precision)

        final_distributions[index] = dist

    return final_distributions


def get_avg_fitness(P):
    """
    Given a distribution of strategies, get the average fitness of the population.
    Payoff is negated for the purpose of optimisation.
    Returns infinity for invalid distributions (also for purpose of optimisation)

    @param P: Distribution of strategies
    @return: Infinity for invalid distribution. Negated average team payoff otherwise
    """
    if sum(P) > 1.0:
        return float('inf')

    if False in [0<=p<=1 for p in P]:
        return float('inf')

    f_novice, f_generalist, f_dropper, f_collector, f_avg = get_fitnesses(P)

    return -1. * f_avg


def get_behaviour_characterisation(P):
    return P


def get_novelty(x, population):
    # Find k nearest neighbours in population and archive
    pass


def get_optimal_distribution():
    """
    Uses differential evolution to find the optimal population of strategies, given a particular payoff function

    @return: An array representing the optimal distribution of strategies, with each element being a float in [0,1]
    and the array having the same length as the number of strategies
    """
    #constraint = LinearConstraint(np.ones(len(strategies)), lb=1, ub=1)
    #bounds = [(0, 1), (0, 1), (0, 1), (0, 1)]
    #res = differential_evolution(get_avg_fitness, bounds, constraints=constraint, maxiter=20000, popsize=200, mutation=0.9, tol=0.0001, seed=random_state, disp=True)
    #final_distribution = [round(x, distribution_precision) for x in res.x]

    options = {'seed': random_state,
               'maxiter': 1000,
               'popsize': 100,
               'bounds': [[0 for i in range(len(strategies))], [1 for i in range(len(strategies))]]}

    # RWG seeding to bootstrap cma
    # (because the landscape is flat)
    samples = sample_distributions(1000)
    min_index = None
    min_value = float('inf')

    for j in range(len(samples)):
        s = samples[j]
        score = get_avg_fitness(s)
        if score < min_value:
            min_value = score
            min_index = j

    es = cma.CMAEvolutionStrategy(samples[min_index], 0.2, options)

    while not es.stop():
        solutions = es.ask()
        es.tell(solutions, [get_avg_fitness(s) for s in solutions])

    final_distribution = [round(x, distribution_precision) for x in es.result.xbest]

    return final_distribution


def calculate_price_of_anarchy(parameter_dictionary, plot_type, axis, team_size=None, slope=None):
    assert None in [team_size, slope], "Can only hold team size or slope size constant. Not both."
    assert team_size is not None or slope is not None, "Must specify either slope or team size to be held constant"
    assert ((plot_type=="slopes" and team_size is not None) or (plot_type=="agents" and slope is not None)), "Plot type must match variables"

    price_list = []

    if plot_type == "slopes":
        xs = slope_list

    elif plot_type == "agents":
        xs = team_list

    for x in xs:
        if plot_type == "slopes":
            generate_constants(parameter_dictionary, team_size=team_size, slope=x)
        elif plot_type == "agents":
            generate_constants(parameter_dictionary, team_size=x, slope=slope)

        optimal_distribution = get_optimal_distribution()
        optimal_payoff = round(-1. * get_avg_fitness(optimal_distribution))

        selfish_distributions = get_many_final_distributions(parameter_dictionary)
        selfish_payoffs = -1. * np.array(list(map(round,list(map(get_avg_fitness, selfish_distributions)))))
        price_of_anarchy = np.mean([optimal_payoff / selfish_payoff for selfish_payoff in selfish_payoffs])
        price_list += [price_of_anarchy]

    axis.plot(xs, price_list, 'o-')

    if plot_type == "slopes":
        axis.set_title(f"{team_size} Agents")
        axis.set(xlabel="Slope", ylabel="Price of Anarchy")

    elif plot_type == "agents":
        axis.set_title(f"Slope {slope}")
        axis.set(xlabel="Num Agents", ylabel="Price of Anarchy")

    # filename = parameter_file.split("/")[-1].strip(".json") + "_price_of_anarchy.png"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot evolutionary dynamics')
    parser.add_argument('--parameters', action="store", dest="parameters")

    parameter_file = parser.parse_args().parameters
    parameter_dictionary = json.loads(open(parameter_file).read())

    '''
    path = "/".join([el for el in parameter_file.split("/")[:-1]]) + "/analysis"

    # Make plots varying team size but holding slope constant (for each slope)
    fig, axs = plt.subplots(1, len(slope_list), sharey=True)
    fig.suptitle("Price of Anarchy vs Team Size")

    for i, slope in enumerate(slope_list):
        calculate_price_of_anarchy(parameter_dictionary, axis=axs[i], plot_type="agents", slope=slope)
        axs[i].label_outer()

    filename = "Price of Anarchy vs Agents"
    plt.savefig(os.path.join(path, filename))

    # Make plots slope but holding team size constant (for each team size)
    fig, axs = plt.subplots(1, len(team_list), sharey=True)
    fig.suptitle("Price of Anarchy vs Slope")

    for i, team_size in enumerate(team_list):
        calculate_price_of_anarchy(parameter_dictionary, axis=axs[i], plot_type="slopes", team_size=team_size)
        axs[i].label_outer()

    filename = "Price of Anarchy vs Slope"
    plt.savefig(os.path.join(path, filename))
    '''

    fig, axs = plt.subplots(2, 1)
    start = time.perf_counter()
    generate_constants(parameter_dictionary, team_size=8, slope=4)
    selfish_distributions = get_many_final_distributions(parameter_dictionary)
    selfish_payoffs = -1. * np.array(list(map(round, list(map(get_avg_fitness, selfish_distributions)))))
    end = time.perf_counter()
    time_taken = end-start
    print(time_taken)


