"""
Modelling slope foraging as a game and plotting the evolutionary dynamics

Strategies:

    Novice- Does nothing. Represents agents that are still learning

    Generalist- Gets resources from the source and brings them back to the nest

    Dropper- Drops resources from the source onto the slope

    Collector- Collects resources from the cache and deposits them at the nest
"""

import matplotlib.pyplot as plt
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
import json
import argparse
import os
import pandas as pd
import numpy as np
import math
import cma
import time
import bisect
import sys
import cProfile
import re

from scipy.optimize import differential_evolution
from scipy.optimize import minimize
from scipy.optimize import Bounds
from scipy.optimize import LinearConstraint
from scipy.integrate import odeint
from functools import partial
from scipy.stats import dirichlet
from itertools import combinations_with_replacement
from io import StringIO
from matplotlib.ticker import PercentFormatter
from helpers import egtsimplex
from glob import glob
from decimal import Decimal,getcontext
getcontext().prec = 10

strategies = ["Novice", "Generalist", "Dropper", "Collector"]
sorted_strategies = strategies[:]
sorted_strategies.sort()
strategy_id = {"Novice": 0, "Generalist": 1, "Dropper": 2, "Collector": 3}
distribution_name = {"[0.0 1.0 0.0 0.0]": "Selfish Equilibrium",
                     "[0.0 0.0 0.5 0.5]": "Cooperative Equilibrium",
                     "[0.0 0.0 0.49 0.51]": "Cooperative Equilibrium"}
N = 256
vermillion = (213/N, 94/N, 0)
bluish_green = (0, 158/N, 115/N)
blue = (0, 114/N, 178/N)
equilibrium_colour = {"Selfish Equilibrium": vermillion,
                      "Cooperative Equilibrium": bluish_green}
cost = {}
payoff_dict = {}
num_agents = 0
generalist_reward = None  # The reward (i.e. resources retrieved) for an individual generalist
specialist_reward = None  # The reward for a dropper and collector pair together
combos = None  # Sorted list of possible strategy combinations for num_agents-1 agents
random_state = None
distribution_precision = 2  # Number of decimal places when expressing the proportion of each strategy in the population
factorial_list = None
multinomial_coefficient_list = {}
partial_cooperation = None

# Formatting
suptitle_font = 55
title_font = 45
label_font = 40
tick_font = 35
legend_font = 35
label_padding = 1.08
linewidth = 5
markersize = 17


def sample_distributions(num_samples=1000, subset_of_population=False):
    """
    Sample a bunch of initial distributions. Each sample is an array of floats summing to 1. Each float represents the
    proportion of the population that uses that particular strategy. If a portion of the population is pre-set to
    Novice (subset_of_population=True) then only sample the remaining strategies

    @param num_samples: Number of distributions to sample
    @param subset_of_population: If true, sample all strategies except Novice
    @return: List of lists of distributions
    """
    if subset_of_population:
        num_strategies = len(strategies) - 1
    else:
        num_strategies = len(strategies)

    sample_format = np.ones(num_strategies)
    return dirichlet.rvs(size=num_samples, alpha=sample_format, random_state=random_state)


def theta(s):
    """
    Step function used to help compute payoff in different slopes
    @param s: Sliding speed (i.e. slope)
    @return: s if the sliding speed is greater than 0 and lower than the saturation point, 0 otherwise
    """
    if s < 0:
        raise RuntimeError("Cannot have a negative value for s")
    elif s == 0:
        return 0
    # Slope saturation- At this sliding speed, resources slide so fast, they arrive right next to (but not on) the nest.
    # Collectors don't have to move to pick them up. Below the saturation point, collectors have to move more
    # because more resources are being dropped
    elif 0 < s < parameter_dictionary["slope_saturation_point"]:
        return s
    else:
        return 0


def generate_constants(parameter_dictionary):
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
    global random_state
    global payoff_dict
    global pair_alignment_probability
    global base_arena_width
    global arena_width
    global factorial_list
    global multinomial_coefficient_list
    global partial_cooperation
    global mutation_probability

    slope = parameter_dictionary["default_slope"]
    num_agents = parameter_dictionary["default_num_agents"]
    generalist_reward = parameter_dictionary["generalist_reward"]
    slope_saturation_point = parameter_dictionary["slope_saturation_point"]
    pair_alignment_probability = parameter_dictionary["pair_alignment_probability"]
    base_arena_width = parameter_dictionary["base_arena_width"]
    arena_width = base_arena_width * num_agents // 2  # Arena width scales linearly with the number of pairs of agents
    partial_cooperation = parameter_dictionary["partial_cooperation"]
    mutation_probability = parameter_dictionary["mutation_probability"]

    for strategy in strategies:
        print(float(np.sum([Decimal(str(prob)) for prob in mutation_probability[strategy.lower()].values()])))
        assert float(np.sum([Decimal(str(prob)) for prob in mutation_probability[strategy.lower()].values()])) == 1.0, "Mutation probabilities do not sum to 1"

    if partial_cooperation == "True":
        partial_cooperation = True

    elif partial_cooperation == "False":
        partial_cooperation = False

    else:
        raise RuntimeError("Invalid setting for parameter: partial_cooperation")

    cost["Novice"] = parameter_dictionary["novice_base_cost"] + parameter_dictionary["novice_cost_multiplier"] * slope
    cost["Generalist"] = parameter_dictionary["generalist_base_cost"] + parameter_dictionary["generalist_cost_multiplier"] * slope
    cost["Dropper"] = parameter_dictionary["dropper_base_cost"] + parameter_dictionary["dropper_cost_multiplier"] * slope
    cost["Collector"] = parameter_dictionary["collector_base_cost"] + parameter_dictionary["collector_cost_multiplier"] * theta(slope)

    if slope >= slope_saturation_point:
        specialist_reward = parameter_dictionary["specialist_reward_high"]
    elif 0 < slope < slope_saturation_point:
        specialist_reward = parameter_dictionary["specialist_reward_low"]
    elif slope == 0:
        specialist_reward = 0
    else:
        raise RuntimeError("Slope must be greater than 0")

    combos = list(combinations_with_replacement(sorted_strategies, num_agents-1))
    random_state = parameter_dictionary["random_state"]
    payoff_dict = {}

    factorial_list = [1 for _ in range(num_agents)]

    for i in range(2, num_agents):
        factorial_list[i] = i * factorial_list[i-1]

    for combo in combos:
        multinomial_coefficient_list[combo] = multinomial_coefficient(combo)

    return


def get_payoff(team, all=False):
    """
    Calculate payoff of every strategy on a team or just first agent

    @param team: List of strings representing the agents' strategies
    @param all: Boolean denoting if returning all payoffs
    @return: The summed payoff (reward - cost) for each agent on the team or payoff for just the first agent
    """

    if team in payoff_dict:
        return payoff_dict[team]

    num_generalists = team.count("Generalist")
    num_droppers = team.count("Dropper")
    num_collectors = team.count("Collector")
    num_spec_pairs = min(num_droppers, num_collectors)

    payoff_from_generalists = (num_generalists * generalist_reward) / num_agents

    if num_collectors != 0:
        # payoff_from_specialists = (num_spec_pairs * specialist_reward) * (num_spec_pairs * pair_alignment_probability) / num_agents
        # payoff_from_specialists = num_spec_pairs * specialist_reward * total_alignment_probability / num_agents

        if pair_alignment_probability == 1.0:
            if partial_cooperation or (num_spec_pairs == num_agents//2):
                payoff_from_specialists = (num_spec_pairs * specialist_reward) / num_agents

            else:
                payoff_from_specialists = 0

        else:
            # How many columns are available for a specialist pair when j-1 Droppers and Collectors have already paired up
            # e.g. If the arena is 8-wide and one pair have matched up, 7 columns are available
            original_available_spaces = arena_width - num_generalists
            available_spaces = [original_available_spaces] + [j for j in range(original_available_spaces, original_available_spaces-num_spec_pairs-1, -1)]

            total_probability = 0

            for i in range(1, num_spec_pairs+1):
                total_probability += pair_alignment_probability * (base_arena_width/available_spaces[0]) * (available_spaces[i-1]/available_spaces[i]) * (num_spec_pairs-i+1)

            payoff_from_specialists = (total_probability * specialist_reward) / num_agents

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

    payoff_dict[team] = payoff
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
    unique_strategies = {"Novice": 0,
                         "Generalist": 0,
                         "Dropper": 0,
                         "Collector": 0}

    for agent_strategy in combo:
        unique_strategies[agent_strategy] += 1

    for key in unique_strategies:
        #denominator *= math.factorial(unique_strategies[key])
        denominator *= factorial_list[unique_strategies[key]]

    # TODO: Can this be more efficient?
    #return math.factorial(n) // denominator
    return factorial_list[n] // denominator


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

    #return product * multinomial_coefficient(teammate_combo)
    return product * multinomial_coefficient_list[teammate_combo]


def get_fitnesses(P):
    combo_probabilities = {}

    for combo in combos:
        combo_probabilities[combo] = get_probability_of_teammate_combo(combo, P)

    f_novice = sum([get_payoff(tuple(["Novice"]) + combo) * combo_probabilities[combo] for combo in combos])
    f_generalist = sum([get_payoff(tuple(["Generalist"]) + combo) * combo_probabilities[combo] for combo in combos])
    f_dropper = sum([get_payoff(tuple(["Dropper"]) + combo) * combo_probabilities[combo] for combo in combos])
    f_collector = sum([get_payoff(tuple(["Collector"]) + combo) * combo_probabilities[combo] for combo in combos])
    f_avg = P[0] * f_novice + P[1] * f_generalist + P[2] * f_dropper + P[3] * f_collector

    return f_novice, f_generalist, f_dropper, f_collector, f_avg


def get_change(P, t=0):
    """
    Function to be passed to ODEINT. Calculates replicator mutator equation for all strategies.
    i.e. Calculates rate of change for each strategy
    e.g.

    Where

    x_dropper = the percentage of the population that are droppers,

    x_dropper' = the rate of increase/decrease of the number of dropper strategies in the population

    f(x_dropper) = the fitness of a dropper in this population (depends on the distribution of other strategies),
    calculated by getting the payoff of a dropper when it is paired with each other strategy then taking the weighted
    sum, where the weight is the probability of being paired with each strategy

    avg_fitness = the average fitness of all other strategies

    Q(j,dropper) = the probability of strategy j mutating into a dropper.

    We then get

    x_dropper' = sum_over_each_strategy_j( f(x_j) * x_j * Q(j,dropper)) - (x_dropper * average_fitness)

    If the Q matrix is the identity matrix then we get the replicator equation:

    x_dropper' = x_dropper(f(x_dropper) - avg_fitness)

    @param P: The distribution of novice, generalist, dropper and collector strategies (in that order) at this time step
    @param t: The current time step
    @return: The derivative of each strategy
    """

    f = get_fitnesses(P)
    f_novice, f_generalist, f_dropper, f_collector, f_avg = f

    '''return np.array([
        P[0] * (f_novice - f_avg),
        P[1] * (f_generalist - f_avg),
        P[2] * (f_dropper - f_avg),
        P[3] * (f_collector - f_avg)
    ])'''
    changes = np.array([
        np.sum([f[j] * P[j] * mutation_probability[strategies[j].lower()]["novice"] for j in range(len(P))]) - (P[0] * f_avg),
        np.sum([f[j] * P[j] * mutation_probability[strategies[j].lower()]["generalist"] for j in range(len(P))]) - (P[1] * f_avg),
        np.sum([f[j] * P[j] * mutation_probability[strategies[j].lower()]["dropper"] for j in range(len(P))]) - (P[2] * f_avg),
        np.sum([f[j] * P[j] * mutation_probability[strategies[j].lower()]["collector"] for j in range(len(P))]) - (P[3] * f_avg),
    ])

    return changes


def get_change_simplex_novices(P, t=0):
    """
    Same as get_change but P is a vector of 3 strategies and holds Novices at 0%
    It calls the normal get_change
    @param P:
    @param t:
    @return:
    """
    new_P = np.concatenate(([0],P))
    return get_change(new_P, t)[1:]


def get_change_simplex_generalists(P, t=0):
    """
    Same as get_change but P is a vector of 3 strategies and holds Generalists at 0%
    It calls the normal get_change
    @param P:
    @param t:
    @return:
    """
    #new_P = np.concatenate(([P[0]], [0], P[1:]))
    #new_P = get_change(new_P, t)
    #return np.concatenate(([new_P[0]], new_P[2:]))

    # Dropper, Novice, Collector (
    new_P = np.concatenate(([P[1]], [0], [P[0]], [P[2]]))
    new_P = get_change(new_P, t)
    return np.concatenate(([new_P[2]], [new_P[0]], [new_P[3]]))


def get_change_simplex_droppers(P, t=0):
    """
    Same as get_change but P is a vector of 3 strategies and holds Droppers at 0%
    It calls the normal get_change
    @param P:
    @param t:
    @return:
    """
    new_P = np.concatenate((P[0:2], [0], [P[2]]))
    new_P = get_change(new_P, t)
    return np.concatenate((new_P[0:2], [new_P[2]]))


def get_change_simplex_collectors(P, t=0):
    """
    Same as get_change but P is a vector of 3 strategies and holds Collectors at 0%
    It calls the normal get_change
    @param P:
    @param t:
    @return:
    """
    #new_P = np.concatenate((P, [0]))
    #new_P = get_change(new_P, t)
    #return new_P[0:3]

    #Novice, Dropper, Generalist
    new_P = np.concatenate(([P[0]], [P[2]], [P[1]], [0]))
    new_P = get_change(new_P, t)
    return np.concatenate(([new_P[0]], [new_P[2]], [new_P[1]]))


def get_change_simplex_cooperators(P, t=0):
    """
    Same as get_change but P is a vector of 3 strategies, Novices, Generalists and Cooperators
    (which is equal Droppers and Collectors)
    It calls the normal get_change
    @param P:
    @param t:
    @return:
    """

    new_P = np.concatenate((P[0:2], [P[2]/2], [P[2]/2]))
    results = get_change(new_P, t)
    return np.concatenate((results[0:2], [results[2] + results[3]]))


def plot_simplex(parameter_dictionary, path):
    fig, axs = plt.subplots(1, 2)
    generate_constants(parameter_dictionary)
    slope = parameter_dictionary["default_slope"]
    fig.suptitle(f'Phase Diagrams for Slope={slope}')

    # Generate plot where Cooperator=Dropper+Collector
    dynamics = egtsimplex.simplex_dynamics(get_change_simplex_cooperators)
    typelabels = ["Novice", "Generalist", "Cooperator"]
    dynamics.plot_simplex(axs[0], typelabels=typelabels)

    # Generate plot where Novices=0
    dynamics = egtsimplex.simplex_dynamics(get_change_simplex_novices)
    typelabels = ["Generalist", "Dropper", "Collector"]
    dynamics.plot_simplex(axs[1], typelabels=typelabels)

    fig.tight_layout()
    fig.subplots_adjust(top=1.2)

    filename = f"simplex_slope={slope}.pdf"
    plt.savefig(os.path.join(path, filename))
    plt.savefig(filename)

    #-------Generate pyramid plot-------#

    #fig, axs = plt.subplots(2, 2, figsize=(10,10))
    generate_constants(parameter_dictionary)
    slope = parameter_dictionary["default_slope"]
    #fig.suptitle(f'Phase Diagrams for Slope={slope} Using All Strategies')
    f_size = (6, 6)


    # Novices=0
    fig, ax = plt.subplots(figsize=f_size)
    dynamics = egtsimplex.simplex_dynamics(get_change_simplex_novices)
    typelabels = ["Generalist", "Dropper", "Collector"]
    #dynamics.plot_simplex(axs[0][0], typelabels=typelabels)
    dynamics.plot_simplex(ax, typelabels=typelabels)
    figname = f"simplex_slope={slope}_pyramid_1.pdf"
    plt.savefig(os.path.join(path, figname), transparent=True)

    # Generalists=0
    fig, ax = plt.subplots(figsize=f_size)
    dynamics = egtsimplex.simplex_dynamics(get_change_simplex_generalists)
    #typelabels = ["Novice", "Dropper", "Collector"]
    typelabels = ["Dropper", "Novice", "Collector"]
    dynamics.plot_simplex(ax, typelabels=typelabels)
    figname = f"simplex_slope={slope}_pyramid_2.pdf"
    plt.savefig(os.path.join(path, figname), transparent=True)


    # Droppers=0
    fig, ax = plt.subplots(figsize=f_size)
    dynamics = egtsimplex.simplex_dynamics(get_change_simplex_droppers)
    typelabels = ["Novice", "Generalist", "Collector"]
    dynamics.plot_simplex(ax, typelabels=typelabels)
    figname = f"simplex_slope={slope}_pyramid_3.pdf"
    plt.savefig(os.path.join(path, figname), transparent=True)

    # Collectors=0
    fig, ax = plt.subplots(figsize=f_size)
    dynamics = egtsimplex.simplex_dynamics(get_change_simplex_collectors)
    #typelabels = ["Novice", "Generalist", "Dropper"]
    typelabels = ["Novice", "Dropper", "Generalist"]
    dynamics.plot_simplex(ax, typelabels=typelabels)
    figname = f"simplex_slope={slope}_pyramid_4.pdf"
    plt.savefig(os.path.join(path, figname), transparent=True)

    #fig.tight_layout()
    #fig.subplots_adjust(top=1.2)

    #filename = f"simplex_slope={slope}_pyramid.pdf"
    #plt.savefig(os.path.join(path, filename))
    #plt.savefig(filename)


def get_distribution_history(parameter_dictionary, P0):
    total_time = parameter_dictionary["total_time"]
    intervals = parameter_dictionary["intervals"]
    t = np.linspace(0, total_time, intervals)
    history = odeint(get_change, P0, t)

    for i in range(len(history)):
        for j in range(len(history[i])):
            history[i][j] = round(history[i][j], distribution_precision)

    return t, history


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
    proportion_of_novices = parameter_dictionary["proportion_of_novices"]

    t = np.linspace(0, total_time, intervals)

    if 0.0 <= proportion_of_novices <= 1.0:
        subset_distributions = (1.0 - proportion_of_novices) * sample_distributions(num_samples, subset_of_population=True)
        novice_list = proportion_of_novices * np.ones(num_samples).reshape(num_samples, 1)
        initial_distributions = np.concatenate((novice_list, subset_distributions), axis=1)

    else:
        initial_distributions = sample_distributions(num_samples)

    final_distributions = [None] * num_samples

    for index in range(len(initial_distributions)):
        P0 = initial_distributions[index]
        all_dist = odeint(get_change, P0, t)
        dist = all_dist[-1]

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


def get_optimal_via_brute_force():

    distribution_fitness = {}
    distribution_fitness['best'] = {'distribution': None,
                                   'score': float('Inf')}
    mask_1 = [0, -1, 0, 1]
    mask_2 = [0, -1, 1, 0]

    def generate_recursively(strategy_distribution):
        if strategy_distribution[1] < 0:
            return

        key = str(strategy_distribution)

        if key in distribution_fitness.keys():
            return

        else:
            distribution_in_percentages = [k/100 for k in strategy_distribution]
            score = get_avg_fitness(distribution_in_percentages)
            distribution_fitness[key] = score

            if score < distribution_fitness['best']['score']:
                distribution_fitness['best']['score'] = score
                distribution_fitness['best']['distribution'] = distribution_in_percentages[:]

            generate_recursively([strategy_distribution[j] + mask_1[j] for j in range(4)])
            generate_recursively([strategy_distribution[j] + mask_2[j] for j in range(4)])

    generate_recursively([0, 100, 0, 0])

    return distribution_fitness['best']['distribution']


def get_optimal_distribution(parameter_dictionary):
    """
    Uses differential evolution to find the optimal population of strategies, given a particular payoff function

    @return: An array representing the optimal distribution of strategies, with each element being a float in [0,1]
    and the array having the same length as the number of strategies
    """

    # Use differential evolution
    '''
    constraint = LinearConstraint(np.ones(len(strategies)), lb=1, ub=1)
    bounds = [(0, 1), (0, 1), (0, 1), (0, 1)]
    res = differential_evolution(get_avg_fitness, bounds, constraints=constraint, maxiter=20000, popsize=200, mutation=0.9, tol=0.0001, seed=random_state, disp=True)
    final_distribution = [round(x, distribution_precision) for x in res.x]
    '''

    # RWG (either as seed for cma or an optimisation approach on its own)
    '''
    samples = sample_distributions(parameter_dictionary["optimisation"]["rwg_samples"])
    min_index = None
    min_value = float('inf')

    for j in range(len(samples)):
        s = [round(item, distribution_precision) for item in samples[j]]
        score = get_avg_fitness(s)
        if score < min_value:
            min_value = score
            min_index = j

    # Use best rwg solution
    final_distribution = [round(item, distribution_precision) for item in samples[min_index]]
    '''

    # Use CMA
    '''
    old_stdout = sys.stdout
    sys.stdout = mystdout = StringIO()

    options = {'seed': random_state,
               'maxiter': parameter_dictionary["optimisation"]["generations"],
               'popsize': parameter_dictionary["optimisation"]["population"],
               'bounds': [[0 for i in range(len(strategies))], [1 for i in range(len(strategies))]]}
    
    es = cma.CMAEvolutionStrategy(samples[min_index], parameter_dictionary["optimisation"]["sigma"], options)

    while not es.stop():
        solutions = es.ask()
        es.tell(solutions, [get_avg_fitness(s) for s in solutions])

    final_distribution = [round(x, distribution_precision) for x in es.result.xbest]

    sys.stdout = old_stdout'''

    # Use brute force
    final_distribution = get_optimal_via_brute_force()

    return final_distribution


def calculate_price_of_anarchy(parameter_dictionary):
    generate_constants(parameter_dictionary)
    use_payoff = parameter_dictionary["use_payoff"]

    if use_payoff == "True":
        use_payoff = True
    elif use_payoff == "False":
        use_payoff = False
    else:
        raise RuntimeError("Invalid value for parameter: use_payoff. Must be True or False")

    if not use_payoff:
        optimal_distribution = get_optimal_distribution(parameter_dictionary)
        optimal_fitness = round(-1. * get_avg_fitness(optimal_distribution))

    else:
        optimal_team = None
        optimal_payoff = float('-Inf')

        for strat in strategies:
            for combo in combos:
                team = tuple([strat]) + combo
                team_payoff = get_payoff(team, all=True)

                if team_payoff > optimal_payoff:
                    optimal_payoff = team_payoff
                    optimal_team = team

        optimal_fitness = optimal_payoff

    selfish_distributions = get_many_final_distributions(parameter_dictionary)
    selfish_fitnesses = -1. * np.array(list(map(round, list(map(get_avg_fitness, selfish_distributions)))))
    selfish_average = np.mean([selfish_fitness for selfish_fitness in selfish_fitnesses])
    price_of_anarchy = optimal_fitness / selfish_average
    print(f"{optimal_fitness} / {selfish_average} = {price_of_anarchy}")

    return price_of_anarchy


def plot_price_of_anarchy(parameter_dictionary, path, prices):
    use_payoff = parameter_dictionary["use_payoff"]
    partial_cooperation = parameter_dictionary["partial_cooperation"]

    fig, ax = plt.subplots(figsize=(30, 15))
    fig.suptitle("Price of Anarchy vs Team Size", fontsize=suptitle_font)

    print("Price of Anarchy vs Team Size")
    slopes = list(prices.keys())
    slopes.sort()

    for slope in [0, 2, 8]:
        team_sizes = list(prices[slope].keys())
        team_sizes.sort()
        data = [prices[slope][n] for n in team_sizes]
        ax.plot(team_sizes, data, marker='o', markersize=markersize, linewidth=linewidth, label=f"Slope {slope}")

    ax.set_xticks([x for x in range(2, 22, 2)])
    ax.set_xlabel("Team Size", fontsize=label_font)
    ax.set_ylabel("Price of Anarchy", fontsize=label_font)
    ax.set_ylim(0.9, 3.1)
    plt.setp(ax.get_xticklabels(), fontsize=tick_font)
    plt.setp(ax.get_yticklabels(), fontsize=tick_font)
    plt.legend(fontsize=legend_font)
    filename = f"price_of_anarchy_vs_team_size_use_payoff={use_payoff}_partial={partial_cooperation}.pdf"
    plt.savefig(os.path.join(path, filename))

    fig, ax = plt.subplots(figsize=(30, 15))
    fig.suptitle("Price of Anarchy vs Slope", fontsize=suptitle_font)

    print("Price of Anarchy vs Slope")
    teams = [2, 4, 8, 20]
    for team in teams:
        data = [prices[slope][team] for slope in slopes]
        ax.plot(slopes, data, marker='o', markersize=markersize, linewidth=linewidth, label=f"{team} Agents")

    ax.set_xlabel("Slope", fontsize=label_font)
    ax.set_ylabel("Price of Anarchy", fontsize=label_font)
    ax.set_ylim(0.9, 3.1)
    plt.setp(ax.get_xticklabels(), fontsize=tick_font)
    plt.setp(ax.get_yticklabels(), fontsize=tick_font)
    plt.legend(fontsize=legend_font)
    filename = f"price_of_anarchy_vs_slope_use_payoff={use_payoff}_partial={partial_cooperation}.pdf"
    plt.savefig(os.path.join(path, filename))


def plot_trend(parameter_dictionary, P0, path):
    #generate_constants(parameter_dictionary, team_size=parameter_dictionary["default_num_agents"], slope=parameter_dictionary["default_slope"])
    generate_constants(parameter_dictionary)
    t, history = get_distribution_history(parameter_dictionary, P0)

    novice, generalist, dropper, collector = history.T
    plt.plot(t, novice, label='Novice')
    plt.plot(t, generalist, label='Generalist')
    plt.plot(t, dropper, label='Dropper')
    plt.plot(t, collector, label='Collector')
    plt.grid()
    plt.ylim(-0.1, 1.1)
    plt.title("Distribution of Strategies Over Time")
    plt.ylabel("Proportion of population")
    plt.xlabel("Time")
    #plt.xticks(fontsize=tick_font)
    #plt.yticks(fontsize=tick_font)
    plt.legend(loc='best')
    filename = f"Trend_{str(P0)}.pdf"
    plt.savefig(os.path.join(path, filename))
    #plt.show()


def get_distribution_frequency(parameter_dictionary):
    generate_constants(parameter_dictionary)
    final_distributions = get_many_final_distributions(parameter_dictionary)
    distribution_dictionary = {}

    for dist in final_distributions:
        key = "[" + " ".join([str(el) for el in abs(dist)]) + "]"

        if key in distribution_dictionary:
            distribution_dictionary[key] += 1
        else:
            distribution_dictionary[key] = 1

    return distribution_dictionary


def plot_distribution_frequency(parameter_dictionary, path, distribution_dictionary, plot_type=None):
    generate_constants(parameter_dictionary)
    total_samples = parameter_dictionary["num_samples"]
    slope = parameter_dictionary["default_slope"]
    novices = parameter_dictionary["proportion_of_novices"]

    if not plot_type:
        fig, ax = plt.subplots(figsize=(30, 15))
        team_size = parameter_dictionary["default_num_agents"]
        slope = parameter_dictionary["default_slope"]
        proportion_of_novices = parameter_dictionary["proportion_of_novices"]
        total_samples = parameter_dictionary["num_samples"]
        pair_alignment_probability = parameter_dictionary["pair_alignment_probability"]

        for i, key in enumerate(distribution_dictionary):
            if key in distribution_name:
                label_name = distribution_name[key]
                colour = equilibrium_colour[label_name]
            else:
                label_name = key
                colour = 'Black'
                distribution_name[key] = key
                equilibrium_colour[key] = colour
                print(label_name)

            ax.bar(i, distribution_dictionary[i][key]/total_samples, color=colour)
            print(distribution_dictionary[key]/total_samples * 100)

        ax.set_xticks(np.arange(len(distribution_dictionary.keys())))
        ax.set_xticklabels([distribution_name[key] for key in distribution_dictionary.keys()], fontsize=tick_font)
        ax.set_title("Frequency of Convergence to Nash Equilibria", fontsize=title_font, y=label_padding)
        ax.set_ylabel("Number of \n Distributions", fontsize=label_font)
        ax.set_xlabel("Nash Equilibria", fontsize=label_font)
        plt.gca().yaxis.set_major_formatter(PercentFormatter(1))

        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(tick_font)

        #ax.legend(loc='best', fontsize=legend_font)
        filename = f"Distribution_Frequency_agents={team_size}_slope={slope}_{pair_alignment_probability}_novices={proportion_of_novices}.pdf"
        plt.savefig(os.path.join(path, filename))
        #plt.show()

    elif plot_type == "slopes":
        # Make plots varying slope but holding team size constant
        fig, ax = plt.subplots(figsize=(30, 15))
        cooperation_rates = [None] * len(distributions_dictionary.keys())
        slopes = list(distribution_dictionary.keys())
        slopes.sort()

        for i, n in enumerate(slopes):
            print(f"Slope: {n}")

            coop_count = 0
            selfish_count = 0

            for j, key in enumerate(distribution_dictionary[n]):
                if key in distribution_name and distribution_name[key] == "Cooperative Equilibrium":
                    coop_count += distribution_dictionary[n][key]

                elif key in distribution_name and distribution_name[key] == "Selfish Equilibrium":
                    selfish_count += distribution_dictionary[n][key]

                else:
                    raise RuntimeError("Unknown Equilibrium")

            cooperation_rates[i] = coop_count / total_samples

        plt.plot(slopes, cooperation_rates, marker='o', markersize=markersize, linewidth=linewidth, color=equilibrium_colour["Cooperative Equilibrium"])

        ax.set_xticks(slopes)
        ax.set_ylim(-0.1, 1.1)
        plt.setp(ax.get_xticklabels(), fontsize=tick_font)
        plt.setp(ax.get_yticklabels(), fontsize=tick_font)

        ax.set_title("Frequency of Cooperation vs Slope", fontsize=title_font, y=label_padding)
        ax.set_ylabel("Frequency of Cooperation", fontsize=label_font)
        ax.set_xlabel("Slope", fontsize=label_font)
        plt.gca().yaxis.set_major_formatter(PercentFormatter(1))

        filename = f"Distribution_Frequency_team_size={num_agents}_all_slopes_partial={partial_cooperation}_novices={novices}.pdf"
        plt.savefig(os.path.join(path, filename))

    elif plot_type == "agents":
        # Make plots varying team size but holding slope constant (for each slope)
        fig, ax = plt.subplots(figsize=(30, 15))
        cooperation_rates = [None] * len(distributions_dictionary.keys())
        team_sizes = list(distribution_dictionary.keys())
        team_sizes.sort()

        #for i,n in enumerate([x for x in range(2,14,2)]):
        for i, n in enumerate(team_sizes):
            print(f"Team Size: {n}")

            coop_count = 0
            selfish_count = 0

            for j, key in enumerate(distribution_dictionary[n]):
                if key in distribution_name and distribution_name[key] == "Cooperative Equilibrium":
                    coop_count += distribution_dictionary[n][key]

                elif key in distribution_name and distribution_name[key] == "Selfish Equilibrium":
                    selfish_count += distribution_dictionary[n][key]

                else:
                    raise RuntimeError("Unknown Equilibrium")

            cooperation_rates[i] = coop_count / total_samples

        plt.plot(team_sizes, cooperation_rates, marker='o', markersize=markersize+2, linewidth=linewidth, color=equilibrium_colour["Cooperative Equilibrium"])

        ax.set_xticks([x for x in range(2,22,2)])
        #ax.set_xticks([x for x in range(2, 14, 2)])
        ax.set_ylim(-0.1, 1.1)
        plt.setp(ax.get_xticklabels(), fontsize=tick_font+10)
        plt.setp(ax.get_yticklabels(), fontsize=tick_font+10)

        ax.set_title("Frequency of Cooperation vs Team Size", fontsize=title_font+10, y=label_padding)
        ax.set_ylabel("Frequency of Cooperation", fontsize=label_font+10)
        ax.set_xlabel("Team Size", fontsize=label_font+10)
        plt.gca().yaxis.set_major_formatter(PercentFormatter(1))

        filename = f"Distribution_Frequency_slope={slope}_all_team_sizes_partial={partial_cooperation}_novices={novices}.pdf"
        plt.savefig(os.path.join(path, filename))


def get_alignment_vs_frequency(parameter_dictionary, path):
    total_samples = parameter_dictionary["num_samples"]

    num_points = 20
    coop_values = [None] * num_points
    alignment_probabilities = np.linspace(0, 100, num_points)
    alignment_probabilities /= 100

    for i, alignment_probability in enumerate(alignment_probabilities):
        parameter_dictionary["pair_alignment_probability"] = alignment_probability
        distribution_dictionary = get_distribution_frequency(parameter_dictionary)

        coop_count = 0
        selfish_count = 0

        for j, key in enumerate(distribution_dictionary):
            if key in distribution_name and distribution_name[key] == "Cooperative Equilibrium":
                coop_count += distribution_dictionary[key]

            elif key in distribution_name and distribution_name[key] == "Selfish Equilibrium":
                selfish_count += distribution_dictionary[key]

            else:
                print(f"Unknown equilibrium: {key}")
                print(f"Alignment probability = {alignment_probability}")
                raise RuntimeError("Unknown Equilibrium")

        coop_values[i] = coop_count/total_samples
        print(coop_values[i])

    # Formatting for paper
    title_font = 95
    label_font = 90
    tick_font = 90
    label_padding = 1.08
    linewidth = 10
    markersize = 35

    fig, ax = plt.subplots(figsize=(45, 25))
    ax.plot(alignment_probabilities, coop_values, marker='o', markersize=markersize, linewidth=linewidth)

    ax.set_title("Impact of Alignment Probability on Degree of Cooperation", fontsize=title_font, y=label_padding)
    ax.set_ylabel("Percentage of\n Cooperative Solutions", fontsize=label_font)
    ax.set_xlabel("Alignment Probability", fontsize=label_font)
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))

    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(tick_font)

    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(tick_font)

    filename = f"Alignment_probability.pdf"
    plt.tight_layout()
    plt.savefig(os.path.join(path, filename))
    # plt.show()


def plot_fitness(parameter_dictionary, path, team_size, slope, plot_name, plot_title):
    generate_constants(parameter_dictionary, team_size=team_size, slope=slope)
    xs = [i/100 for i in range(0, 105, 5)]
    fit_generalists = [None for _ in range(len(xs))]
    fit_droppers = [None for _ in range(len(xs))]
    fit_collectors = [None for _ in range(len(xs))]
    fit_avg = [None for _ in range(len(xs))]

    # Get fitness of each strategy as the ratio of cooperator to generalist increases
    for i, x in enumerate(xs):
        p_generalists = 1.0-x
        p_droppers = p_collectors = x/2
        p = [0.0, p_generalists, p_droppers, p_collectors]
        f_n, f_g, f_d, f_c, f_avg = get_fitnesses(p)
        fit_generalists[i] = f_g
        fit_droppers[i] = f_d
        fit_collectors[i] = f_c
        fit_avg[i] = f_avg

    fig, ax = plt.subplots()
    ax.plot(xs, fit_generalists, label='Generalist')
    ax.plot(xs, fit_droppers, label='Dropper')
    ax.plot(xs, fit_collectors, label='Collector')
    ax.plot(xs, fit_avg, label='Average')
    ax.set_title(plot_title)
    ax.set_xlabel("Ratio of cooperators to generalists")
    ax.set_ylabel("Fitness")
    ax.set_ylim(0, 10000)
    ax.grid()
    ax.legend(loc='best')
    filename = f"ratio_vs_fitness_{plot_name}.pdf"
    plt.savefig(os.path.join(path, filename))


    # For every combination of coop and generalist
    # Calculate fitnesses
    # Store fitnesses in a list
    # Plot each list


def plot_probability(parameter_dictionary, path, slope, population_distribution):
    probs = [None] * len(team_list)

    for i, team_size in enumerate(team_list):
        generate_constants(parameter_dictionary, team_size=team_size, slope=slope)
        probability_of_collector = 0

        for combo in combos:
            if "Collector" in combo:
                probability_of_collector += get_probability_of_teammate_combo(combo, population_distribution)

        probs[i] = probability_of_collector

    fig, ax = plt.subplots()
    ax.plot(team_list, probs, marker='o', markersize=markersize, linewidth=linewidth)
    ax.set_ylim(0, 1)
    ax.set_title("Probability of A Collector Team-mate")
    ax.set_xlabel("Team Size")
    ax.set_ylabel("Probability")
    filename = f"probability_vs_team_size.pdf"
    plt.savefig(os.path.join(path, filename))


def plot_payoff_vs_agents(parameter_dictionary, path, max_team_size):
    payoff_list = [None] * (max_team_size//2)
    alignment_probability = parameter_dictionary["pair_alignment_probability"]

    for i in range(1, max_team_size//2 + 1):
        parameter_dictionary["default_num_agents"] = i*2
        generate_constants(parameter_dictionary)
        team = tuple(["Dropper"] * i + ["Collector"] * i)
        payoff_list[i-1] = get_payoff(team=team, all=True) / (i*2)

    fig, ax = plt.subplots(figsize=(16, 8))
    xs = [x for x in range(2, max_team_size + 2, 2)]
    plt.plot(xs, payoff_list, marker='o', markersize=markersize-2, linewidth=linewidth)
    #ax.set_xlim(xs[0], xs[-1])
    plt.setp(ax.get_xticklabels(), fontsize=tick_font-20)
    plt.setp(ax.get_yticklabels(), fontsize=tick_font-20)
    ax.set_ylim(0, 25000)

    if alignment_probability != 1.0:
        ax.set_title("Scalability of Performance When Agents Fail to Coordinate", fontsize=title_font-20)
    else:
        ax.set_title("Scalability of Performance When Agents Always Coordinate", fontsize=title_font-20)

    ax.set_xlabel("Team Size", fontsize=label_font-20)
    ax.set_ylabel("Payoff per Agent", fontsize=label_font-20)
    filename = f"payoff_vs_agents_{pair_alignment_probability}.pdf"
    plt.savefig(os.path.join(path, filename))
    #plt.show()


def plot_payoff_vs_agents_combined(parameter_dictionary, path, max_team_size, alignments):
    fig, ax = plt.subplots(figsize=(16, 8))
    xs = [x for x in range(2, max_team_size + 2, 2)]

    for alignment in alignments:
        payoff_list = [None] * (max_team_size//2)
        parameter_dictionary["pair_alignment_probability"] = alignment
        alignment_probability = parameter_dictionary["pair_alignment_probability"]

        for i in range(1, max_team_size//2 + 1):
            parameter_dictionary["default_num_agents"] = i*2
            generate_constants(parameter_dictionary)
            team = tuple(["Dropper"] * i + ["Collector"] * i)
            payoff_list[i-1] = get_payoff(team=team, all=True) / (i*2)

        plt.plot(xs, payoff_list, marker='o', markersize=markersize-2, linewidth=linewidth, label=alignment)
        #ax.set_xlim(xs[0], xs[-1])

    plt.legend(loc='upper right', fontsize=legend_font-10)
    plt.setp(ax.get_xticklabels(), fontsize=tick_font-10)
    plt.setp(ax.get_yticklabels(), fontsize=tick_font-10)
    ax.set_ylim(-1000, 25000)
    ax.set_title("Scalability of Performance with Varying Alignment Challenge", fontsize=title_font-10, pad=30)
    ax.set_xlabel("Team Size", fontsize=label_font-10)
    ax.set_ylabel("Payoff per Agent", fontsize=label_font-10)
    filename = f"payoff_vs_agents_combined.pdf"
    plt.savefig(os.path.join(path, filename))
    #plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot evolutionary dynamics')
    parser.add_argument('--parameters', action="store", dest="parameters")
    parser.add_argument('--function', action="store", dest="function")
    parser.add_argument('--args_to_pass', action="store", dest="args_to_pass")
    parser.add_argument('--num_agents', action="store", dest="num_agents")
    parser.add_argument('--slope', action="store", dest="slope")
    parser.add_argument('--alignments', action="store", dest="alignments")

    parameter_file = parser.parse_args().parameters
    function = parser.parse_args().function
    args_to_pass = parser.parse_args().args_to_pass
    num_agents = parser.parse_args().num_agents
    slope = parser.parse_args().slope
    alignments = parser.parse_args().alignments
    parameter_dictionary = json.loads(open(parameter_file).read())
    path = "/".join([el for el in parameter_file.split("/")[:-1]]) + "/analysis"

    # Profiling
    if function == "profile":
        profiling_file = os.path.join(path, "profiling_v2_dynamic_programming.csv")
        pstats = cProfile.run('plot_distribution_frequency(parameter_dictionary, path)', profiling_file, 1)
        p = pstats.Stats(profiling_file)
        p.sort_stats('cumtime').print_stats(20)
        p.sort_stats('tottime').print_stats(20)

    elif function == "trend":
        args_to_pass = [float(x) for x in args_to_pass.strip("[]").split(",")]
        plot_trend(parameter_dictionary, args_to_pass, path)

    elif function == "get_distribution":
        distribution_dictionary = get_distribution_frequency(parameter_dictionary)
        results_file = os.path.join(path, args_to_pass)
        output_file = open(results_file, "w")
        json.dump(distribution_dictionary, output_file, indent=4)
        output_file.close()

    elif function == "plot_distribution":
        distribution_file = args_to_pass
        dict_path = os.path.join(path, distribution_file)
        distribution_dictionary = json.loads(open(dict_path).read())
        plot_distribution_frequency(parameter_dictionary, path, distribution_dictionary)

    elif function == "get_distribution_agents":
        directory = args_to_pass
        num_agents = int(num_agents)
        parameter_dictionary["default_num_agents"] = num_agents
        distribution_dictionary = get_distribution_frequency(parameter_dictionary)
        results_path = os.path.join(path, directory)

        if not os.path.exists(results_path):
            os.makedirs(results_path)

        results_file = os.path.join(results_path, f"{num_agents}.json")
        output_file = open(results_file, "w")
        json.dump(distribution_dictionary, output_file, indent=4)
        output_file.close()

    elif function == "get_distribution_slopes":
        directory = args_to_pass
        slope = int(slope)
        parameter_dictionary["default_slope"] = slope
        distribution_dictionary = get_distribution_frequency(parameter_dictionary)
        results_path = os.path.join(path, directory)

        if not os.path.exists(results_path):
            os.makedirs(results_path)

        results_file = os.path.join(results_path, f"slope={slope}.json")
        output_file = open(results_file, "w")
        json.dump(distribution_dictionary, output_file, indent=4)
        output_file.close()

    elif function == "plot_distribution_agents":
        directory = args_to_pass
        data_path = os.path.join(path, directory)
        distributions_dictionary = {}

        for filename in glob(f"{data_path}/*.json"):
            stripped_name = filename.split("/")[-1].strip(".json")
            if "=" not in stripped_name:
                num_agents = int(stripped_name)
                dict_path = os.path.join(data_path, filename)
                distributions_dictionary[num_agents] = json.loads(open(dict_path).read())

        plot_distribution_frequency(parameter_dictionary, path, distributions_dictionary, plot_type="agents")

    elif function == "plot_distribution_slopes":
        directory = args_to_pass
        data_path = os.path.join(path, directory)
        distributions_dictionary = {}

        for filename in glob(f"{data_path}/*=*.json"):
            slope = int(filename.split("/")[-1].strip(".json").split("=")[-1])
            dict_path = os.path.join(data_path, filename)
            distributions_dictionary[slope] = json.loads(open(dict_path).read())

        plot_distribution_frequency(parameter_dictionary, path, distributions_dictionary, plot_type="slopes")

    elif function == "plot_alignment_frequency":
        get_alignment_vs_frequency(parameter_dictionary, path)

    elif function == "plot_payoff_vs_agents":
        max_team_size = int(args_to_pass)
        plot_payoff_vs_agents(parameter_dictionary, path, max_team_size)

    elif function == "plot_payoff_vs_agents_combined":
        max_team_size = int(args_to_pass)

        if not alignments:
            raise RuntimeError("Did not pass a list of alignments")

        alignments = [float(a) for a in alignments.strip('[]').split(',')]

        plot_payoff_vs_agents_combined(parameter_dictionary, path, max_team_size, alignments)

    elif function == "calculate_price_of_anarchy":
        num_agents = int(num_agents)
        slope = int(slope)
        parameter_dictionary["default_num_agents"] = num_agents
        parameter_dictionary["default_slope"] = slope
        price_of_anarchy = {"poa": calculate_price_of_anarchy(parameter_dictionary),
                            "agents": num_agents,
                            "slope": slope}

        directory = args_to_pass
        results_path = os.path.join(path, directory)

        if not os.path.exists(results_path):
            os.makedirs(results_path)

        results_file = os.path.join(results_path, f"agents={num_agents}_slope={slope}.json")
        output_file = open(results_file, "w")
        json.dump(price_of_anarchy, output_file, indent=4)
        output_file.close()

    elif function == "plot_price_of_anarchy":
        directory = args_to_pass
        data_path = os.path.join(path, directory)
        prices = {}

        for filename in glob(f"{data_path}/*.json"):
            data = json.loads(open(filename).read())
            num_agents = data["agents"]
            slope = data["slope"]
            poa = data["poa"]

            if slope not in prices:
                prices[slope] = {}

            prices[slope][num_agents] = poa

        plot_price_of_anarchy(parameter_dictionary, path, prices)

    elif function == "plot_simplex":
        plot_simplex(parameter_dictionary, path)



    #
    #plot_distribution_frequency(parameter_dictionary, path, plot_type="slopes")

    #plot_simplex(parameter_dictionary)
    #plot_simplex(parameter_dictionary, team_size=4)

    #plot_fitness(parameter_dictionary, path, team_size=2, slope=8, plot_name="slope_8", plot_title="Fitness vs Strategy Ratio when Slope=8")
    #plot_fitness(parameter_dictionary, path, team_size=2, slope=4, plot_name="slope_4", plot_title="Fitness vs Strategy Ratio when Slope=4")
    #plot_fitness(parameter_dictionary, path, team_size=2, slope=1, plot_name="slope_1", plot_title="Fitness vs Strategy Ratio when Slope=1")
    #plot_fitness(parameter_dictionary, path, team_size=2, slope=2, plot_name="slope_2", plot_title="Fitness vs Strategy Ratio when Slope=2")

    #plot_probability(parameter_dictionary, path, slope=4, population_distribution=[0.25, 0.25, 0.25, 0.25])
    #plot_fitness(parameter_dictionary, path, team_size=8, slope=4, plot_name="slope_4_team_8", plot_title="Fitness vs Strategy Ratio when Slope=4 and Team=8")

    #plot_price_of_anarchy(parameter_dictionary, path)





