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
import sys

from scipy.optimize import differential_evolution
from scipy.optimize import minimize
from scipy.optimize import Bounds
from scipy.optimize import LinearConstraint
from scipy.integrate import odeint
from functools import partial
from scipy.stats import dirichlet
from itertools import combinations_with_replacement
from io import StringIO

strategies = ["Novice", "Generalist", "Dropper", "Collector"]
strategy_id = {"Novice": 0, "Generalist": 1, "Dropper": 2, "Collector": 3}
distribution_name = {"[0.0 1.0 0.0 0.0]": "Selfish Equilibrium",
                     "[0.0 0.0 0.5 0.5]": "Cooperative Equilibrium"}
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
slope_list = [i for i in range(9)]
team_list = [i for i in range(2, 9)]
random_state = None
distribution_precision = 2  # Number of decimal places when expressing the proportion of each strategy in the population

# Formatting
suptitle_font = 40
title_font = 30
label_font = 26
tick_font = 20
legend_font = 20
label_padding = 1.08
linewidth = 5
markersize = 17

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
    global random_state
    global payoff_dict

    if slope is None:
        slope = parameter_dictionary["default_slope"]

    cost["Novice"] = parameter_dictionary["novice_base_cost"] + \
                     parameter_dictionary["novice_cost_multiplier"] * slope

    cost["Generalist"] = parameter_dictionary["generalist_base_cost"] + \
                         parameter_dictionary["generalist_cost_multiplier"] * slope

    cost["Dropper"] = parameter_dictionary["dropper_base_cost"] + \
                      parameter_dictionary["dropper_cost_multiplier"] * slope

    cost["Collector"] = parameter_dictionary["collector_base_cost"] + \
                        parameter_dictionary["collector_cost_multiplier"] * theta(slope)

    if team_size is None:
        num_agents = parameter_dictionary["default_num_agents"]
    else:
        num_agents = team_size

    generalist_reward = parameter_dictionary["generalist_reward"]
    specialist_reward = parameter_dictionary["specialist_base_reward"] * slope

    sorted_strategies = strategies[:]
    sorted_strategies.sort()

    combos = list(combinations_with_replacement(sorted_strategies, num_agents-1))
    random_state = parameter_dictionary["random_state"]
    payoff_dict = {}

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

    f_novice = sum([get_payoff(tuple(["Novice"]) + combo) * combo_probabilities[combo] for combo in combos])
    f_generalist = sum([get_payoff(tuple(["Generalist"]) + combo) * combo_probabilities[combo] for combo in combos])
    f_dropper = sum([get_payoff(tuple(["Dropper"]) + combo) * combo_probabilities[combo] for combo in combos])
    f_collector = sum([get_payoff(tuple(["Collector"]) + combo) * combo_probabilities[combo] for combo in combos])
    f_avg = P[0] * f_novice + P[1] * f_generalist + P[2] * f_dropper + P[3] * f_collector

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


def get_distribution_history(parameter_dictionary, P0):
    total_time = parameter_dictionary["total_time"]
    intervals = parameter_dictionary["intervals"]
    t = np.linspace(0, intervals, total_time)
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


def get_optimal_via_brute_force():

    distribution_fitness = {}
    distribution_fitness['best'] = {'distribution': None,
                                   'score': float('Inf')}
    mask_1 = [-1, 1, 0, 0]
    mask_2 = [-1, 0, 1, 0]
    mask_3 = [-1, 0, 0, 1]

    def generate_recursively(strategy_distribution):
        if strategy_distribution[0] < 0:
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
            generate_recursively([strategy_distribution[j] + mask_3[j] for j in range(4)])

    generate_recursively([100, 0, 0, 0])
    #print(distribution_fitness['best'])
    #print(distribution_fitness['[0, 0, 50, 50]'])
    #print("Did it work?")

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
        print(f'{plot_type}: {x}')
        if plot_type == "slopes":
            if prices_of_anarchy[x][team_size]:
                price_list += [prices_of_anarchy[x][team_size]]
                continue

            generate_constants(parameter_dictionary, team_size=team_size, slope=x)

        elif plot_type == "agents":
            if prices_of_anarchy[slope][x]:
                price_list += [prices_of_anarchy[slope][x]]
                continue

            generate_constants(parameter_dictionary, team_size=x, slope=slope)

        optimal_distribution = get_optimal_distribution(parameter_dictionary)
        optimal_payoff = round(-1. * get_avg_fitness(optimal_distribution))

        selfish_distributions = get_many_final_distributions(parameter_dictionary)
        selfish_payoffs = -1. * np.array(list(map(round,list(map(get_avg_fitness, selfish_distributions)))))
        price_of_anarchy = np.mean([optimal_payoff / selfish_payoff for selfish_payoff in selfish_payoffs])
        price_list += [price_of_anarchy]
        '''print(f"Optimal distribution: {optimal_distribution}")
        print(f"Optimal payoff = {optimal_payoff}")
        print(f"Avg team payoff = {round(optimal_payoff / price_of_anarchy)}")
        print(f"Price of Anarchy = {round(price_of_anarchy, 2)}")
        print()
        print(f"Dropper-Collector Payoff = {-1.0 * get_avg_fitness([0,0,0.5,0.5])}")
        print(f"Generalist Payoff = {-1.0 * get_avg_fitness([0, 1.0, 0, 0])}")
        print(f"Generalist 10% Payoff = {-1.0 * get_avg_fitness([0, 0.1, 0.45, 0.45])}")
        print(f"Generalist 20% Payoff = {-1.0 * get_avg_fitness([0, 0.2, 0.4, 0.4])}")
        print(f"Generalist 30% Payoff = {-1.0 * get_avg_fitness([0, 0.3, 0.35, 0.35])}")
        print(f"Generalist 40% Payoff = {-1.0 * get_avg_fitness([0, 0.4, 0.3, 0.3])}")
        print(f"Generalist 50% Payoff = {-1.0 * get_avg_fitness([0, 0.5, 0.25, 0.25])}")
        print()'''

        if plot_type == "slopes":
            prices_of_anarchy[x][team_size] = price_of_anarchy

        elif plot_type == "agents":
            prices_of_anarchy[slope][x] = price_of_anarchy

    axis.plot(xs, price_list, 'o-', markersize=markersize, linewidth=linewidth, color=blue)

    if plot_type == "slopes":
        axis.set_title(f"{team_size} Agents", fontsize=title_font)
        axis.set_xlabel("Slope", fontsize=label_font)
        axis.set_ylabel("Price of Anarchy", fontsize=label_font)
        #axis.set(xlabel="Slope", ylabel="Price of Anarchy")

    elif plot_type == "agents":
        axis.set_title(f"Slope {slope}", fontsize=title_font)
        axis.set_xlabel("Num Agents", fontsize=label_font)
        axis.set_ylabel("Price of Anarchy", fontsize=label_font)
        #axis.set(xlabel="Num Agents", ylabel="Price of Anarchy")

    # filename = parameter_file.split("/")[-1].strip(".json") + "_price_of_anarchy.png"


def plot_price_of_anarchy(parameter_dictionary, path):
    cols = 3

    # Make plots varying team size but holding slope constant (for each slope)
    rows = math.ceil(len(slope_list) / cols)
    fig, axs = plt.subplots(rows, cols, sharey=True, figsize=(30, 15))
    fig.suptitle("Price of Anarchy vs Team Size", fontsize=suptitle_font)

    print("Price of Anarchy vs Team Size")
    for i, slope in enumerate(slope_list):
        print(f"Slope {slope}")
        row_id = i // rows
        col_id = i % cols
        calculate_price_of_anarchy(parameter_dictionary, axis=axs[row_id][col_id], plot_type="agents", slope=slope)
        axs[row_id][col_id].label_outer()

        for tick in axs[row_id][col_id].xaxis.get_major_ticks():
            tick.label.set_fontsize(tick_font)

        for tick in axs[row_id][col_id].yaxis.get_major_ticks():
            tick.label.set_fontsize(tick_font)

    filename = "Price of Anarchy vs Team Size"
    #plt.rcParams.update({'font.size': 18})
    plt.savefig(os.path.join(path, filename))

    # Make plots slope but holding team size constant (for each team size)
    rows = math.ceil(len(team_list) / cols)
    fig, axs = plt.subplots(rows, cols, sharey=True, figsize=(30, 15))
    fig.suptitle("Price of Anarchy vs Slope")

    print("Price of Anarchy vs Slope")
    for i, team_size in enumerate(team_list):
        print(f"Team size {team_size}")
        row_id = i // rows
        col_id = i % cols
        # if team_size == 2:
        calculate_price_of_anarchy(parameter_dictionary, axis=axs[row_id][col_id], plot_type="slopes",
                                   team_size=team_size)
        axs[row_id][col_id].label_outer()

        for tick in axs[row_id][col_id].xaxis.get_major_ticks():
            tick.label.set_fontsize(tick_font)

        for tick in axs[row_id][col_id].yaxis.get_major_ticks():
            tick.label.set_fontsize(tick_font)

    filename = "Price of Anarchy vs Slope"
    #plt.rcParams.update({'font.size': 18})
    plt.savefig(os.path.join(path, filename))


def plot_trend(parameter_dictionary, P0, path):
    generate_constants(parameter_dictionary, team_size=parameter_dictionary["default_num_agents"], slope=parameter_dictionary["default_slope"])
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
    # plt.show()


def get_distribution_frequency(parameter_dictionary, team_size=None, slope=None):
    if team_size is None:
        team_size = parameter_dictionary["default_num_agents"]

    if slope is None:
        slope = parameter_dictionary["default_slope"]

    generate_constants(parameter_dictionary, team_size, slope)
    final_distributions = get_many_final_distributions(parameter_dictionary)
    distribution_dictionary = {}

    for dist in final_distributions:
        key = "[" + " ".join([str(el) for el in abs(dist)]) + "]"

        if key in distribution_dictionary:
            distribution_dictionary[key] += 1
        else:
            distribution_dictionary[key] = 1

    return distribution_dictionary


def plot_distribution_frequency(parameter_dictionary, path, plot_type=None):
    if not plot_type:
        fig, ax = plt.subplots(figsize=(30, 15))
        distribution_dictionary = get_distribution_frequency(parameter_dictionary)

        for i, key in enumerate(distribution_dictionary):
            label_name = distribution_name[key]
            colour = equilibrium_colour[label_name]
            ax.bar(i, distribution_dictionary[key], color=colour)

        ax.set_xticks(np.arange(len(distribution_dictionary.keys())))
        ax.set_xticklabels([distribution_name[key] for key in distribution_dictionary.keys()], fontsize=tick_font)
        ax.set_title("Frequency of Convergence to Nash Equilibria", fontsize=title_font, y=label_padding)
        ax.set_ylabel("Number of \n Distributions", fontsize=label_font)
        ax.set_xlabel("Nash Equilibria", fontsize=label_font)

        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(tick_font)

        #ax.legend(loc='best', fontsize=legend_font)
        slope = parameter_dictionary["default_slope"]
        filename = f"Distribution_Frequency_agents={num_agents}_slope={slope}.pdf"
        plt.savefig(os.path.join(path, filename))
        #plt.show()

    elif plot_type == "slopes":
        cols = 3

        # Make plots varying team size but holding slope constant (for each slope)
        rows = math.ceil(len(slope_list) / cols)
        fig, axs = plt.subplots(rows, cols, sharey=True, figsize=(30, 15))
        fig.suptitle("Convergence to Nash Equilibria vs Slope", fontsize=suptitle_font)

        for i, slope in enumerate(slope_list):
            print(f"Slope: {slope}")
            row_id = i // rows
            col_id = i % cols
            ax = axs[row_id][col_id]
            distribution_dictionary = get_distribution_frequency(parameter_dictionary, slope=slope)

            for j, key in enumerate(distribution_dictionary):
                label_name = distribution_name[key]
                colour = equilibrium_colour[label_name]
                ax.bar(j, distribution_dictionary[key], color=colour)

            ax.set_xticks(np.arange(len(distribution_dictionary.keys())))
            ax.set_xticklabels([distribution_name[key] for key in distribution_dictionary.keys()], fontsize=tick_font)
            ax.set_title(f"Slope {slope}", fontsize=title_font, y=label_padding)
            ax.set_ylabel("Number of \n Distributions", fontsize=label_font)
            ax.set_xlabel("Nash Equilibria", fontsize=label_font)

            for tick in ax.yaxis.get_major_ticks():
                tick.label.set_fontsize(tick_font)

            #ax.legend(loc='best', fontsize=legend_font)
            ax.label_outer()
            filename = f"Distribution_Frequency_agents={num_agents}_all_slopes.pdf"
            plt.savefig(os.path.join(path, filename))

    elif plot_type == "agents":
        cols = 3

        # Make plots varying team size but holding slope constant (for each slope)
        rows = math.ceil(len(slope_list) / cols)
        fig, axs = plt.subplots(rows, cols, sharey=True, figsize=(30, 15))
        fig.suptitle("Convergence to Nash Equilibria vs Team Size")

        for i, team_size in enumerate(team_list):
            print(f"Team Size: {team_size}")
            row_id = i // rows
            col_id = i % cols
            ax = axs[row_id][col_id]
            distribution_dictionary = get_distribution_frequency(parameter_dictionary, team_size=team_size)

            for j, key in enumerate(distribution_dictionary):
                label_name = distribution_name[key]
                colour = equilibrium_colour[label_name]
                ax.bar(j, distribution_dictionary[key], label=label_name, color=colour)

            ax.set_xticks(np.arange(len(distribution_dictionary.keys())))
            ax.set_xticklabels([distribution_name[key] for key in distribution_dictionary.keys()], fontsize=tick_font)
            ax.set_title(f"{team_size} Agents", fontsize=title_font, y=label_padding)
            ax.set_ylabel("Number of \n Distributions", fontsize=label_font)
            ax.set_xlabel("Nash Equilibria", fontsize=label_font)

            for tick in ax.yaxis.get_major_ticks():
                tick.label.set_fontsize(tick_font)

            #ax.legend(loc='best', fontsize=legend_font)
            ax.label_outer()
            slope = parameter_dictionary["default_slope"]
            filename = f"Distribution_Frequency_slope={slope}_all_team_sizes.pdf"
            plt.savefig(os.path.join(path, filename))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot evolutionary dynamics')
    parser.add_argument('--parameters', action="store", dest="parameters")

    parameter_file = parser.parse_args().parameters
    parameter_dictionary = json.loads(open(parameter_file).read())

    path = "/".join([el for el in parameter_file.split("/")[:-1]]) + "/analysis"

    #plot_trend(parameter_dictionary, [0.85, 0.05, 0.05, 0.05], path)
    #plot_trend(parameter_dictionary, [0.7, 0.2, 0.05, 0.05], path)

    #plot_distribution_frequency(parameter_dictionary, path)
    #plot_distribution_frequency(parameter_dictionary, path, plot_type="slopes")
    #plot_distribution_frequency(parameter_dictionary, path, plot_type="agents")

    plot_price_of_anarchy(parameter_dictionary, path)




