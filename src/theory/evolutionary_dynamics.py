"""
Modelling slope foraging as a game and plotting the evolutionary dynamics

Strategies:
Novice-
Generalist-
Dropper-
Collector-
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

strategy = ["Novice", "Generalist", "Dropper", "Collector"]


def sample_distributions(num_samples=1000):
    num_strategies = len(strategy)
    sample_format = np.ones(num_strategies)
    return dirichlet.rvs(size=num_samples, alpha=sample_format)


def theta(s):
    """
    Step function used to help compute payoff in different slopes
    @param s: Sliding speed (i.e. slope)
    @return:
    """
    if s == 0:
        return 0
    elif s > 0:
        return 1


def get_payoff_matrix(parameter_file):
    """
    Gets relevant values from json file and prepares the payoff matrix or matrices
    @param parameter_file: Path to json file
    @return: Payoff matrix for each pair of strategies if it is an equilibrium experiment. List of payoff matrices if
    it is a price of anarchy experiment
    """

    parameter_dictionary = json.loads(open(parameter_file).read())

    e = parameter_dictionary["e"]
    g = parameter_dictionary["g"]
    b = parameter_dictionary["b"]
    f = parameter_dictionary["f"]
    c = parameter_dictionary["c"]

    if parameter_dictionary["experiment_type"] == "equilibrium":
        slope_list = [parameter_dictionary["equilibrium"]["s"]] # Sliding speed

    elif parameter_dictionary["experiment_type"] == "price_of_anarchy":
        slope_list = parameter_dictionary["price_of_anarchy"]["slopes"]

    payoff_matrices = []

    for s in slope_list:
        # Payoff for each strategy when it is paired with another strategy (when there is a slope)
        payoff = {"Novice": {"Novice": s*theta(s)*f + (1-theta(s))*b,
                             "Generalist": s*theta(s)*f + (1-theta(s))*b,
                             "Dropper": s*theta(s)*f + (1-theta(s))*b,
                             "Collector": s*theta(s)*f + (1-theta(s))*b},

                  "Generalist": {"Novice": g - s*c,
                                 "Generalist": g - s*c,
                                 "Dropper": g - s*c,
                                 "Collector": g - s*c},

                  "Dropper": {"Novice": s*theta(s)*f + (1-theta(s))*b,
                              "Generalist": s*theta(s)*f + (1-theta(s))*b,
                              "Dropper": s*theta(s)*f + (1-theta(s))*b,
                              "Collector": s*theta(s)*e + (1-theta(s))*b},

                  "Collector": {"Novice": b,
                                "Generalist": b,
                                "Dropper": s*theta(s)*e + (1-theta(s))*b,
                                "Collector": b}}

        payoff_matrices += [payoff]

    if len(payoff_matrices) == 1:
        return payoff_matrices[0]
    else:
        return payoff_matrices


def get_change(P, t=0):
    """
    Function to be passed to ODEINT. Calculates replicator dynamic equation for all strategies.
    i.e. Calculates rate of change for each strategy
    e.g.

    Where

    x_dropper = the percentage of the population that are droppers,

    x_dropper' = the rate of increase/decrease of the number of dropper strategies in the population

    f(x_dropper) = the fitness of a droppers in this population (depends on the distribution of other strategies),
    calculated by getting the payoff of a dropper when it is paired with each other strategy then taking the weighted
    sum, where the weight is the probability of being paired with each strategy

    avg_fitness = the average fitness of all other strategies

    We then get

    x_dropper' = x_dropper(f(x_dropper) - avg_fitness)

    @param P: The distribution of novice, generalist, dropper and collector strategies (in that order) at this time step
    @param t: The current time step
    @return: The derivative of each strategy
    """

    f_novice = sum([P[i] * payoff["Novice"][strategy[i]] for i in range(len(strategy))])
    f_generalist = sum([P[i] * payoff["Generalist"][strategy[i]] for i in range(len(strategy))])
    f_dropper = sum([P[i] * payoff["Dropper"][strategy[i]] for i in range(len(strategy))])
    f_collector = sum([P[i] * payoff["Collector"][strategy[i]] for i in range(len(strategy))])
    f_avg = P[0] * f_novice + P[1] * f_generalist + P[2] * f_dropper + P[3] * f_collector

    return np.array([
        P[0] * (f_novice - f_avg),
        P[1] * (f_generalist - f_avg),
        P[2] * (f_dropper - f_avg),
        P[3] * (f_collector - f_avg)
    ])


def get_change_centralised(P, t=0):
    f_novice = sum([P[i] * (payoff["Novice"][strategy[i]] + payoff[strategy[i]]["Novice"]) for i in range(len(strategy))])
    f_generalist = sum([P[i] * (payoff["Generalist"][strategy[i]] + payoff[strategy[i]]["Generalist"]) for i in range(len(strategy))])
    f_dropper = sum([P[i] * (payoff["Dropper"][strategy[i]] + payoff[strategy[i]]["Dropper"]) for i in range(len(strategy))])
    f_collector = sum([P[i] * (payoff["Collector"][strategy[i]] + payoff[strategy[i]]["Collector"]) for i in range(len(strategy))])
    f_avg = P[0] * f_novice + P[1] * f_generalist + P[2] * f_dropper + P[3] * f_collector

    return np.array([
        P[0] * (f_novice - f_avg),
        P[1] * (f_generalist - f_avg),
        P[2] * (f_dropper - f_avg),
        P[3] * (f_collector - f_avg)
    ])


def get_final_distribution(parameter_file, payoff_local, setup):
    parameter_dictionary = json.loads(open(parameter_file).read())

    global payoff
    payoff = payoff_local

    #total_time = parameter_dictionary["equilibrium"]["total_time"]

    t = np.linspace(0, 50, 1000)

    P0 = np.array([parameter_dictionary["equilibrium"]["initial_distribution"]["Novice"],
         parameter_dictionary["equilibrium"]["initial_distribution"]["Generalist"],
         parameter_dictionary["equilibrium"]["initial_distribution"]["Dropper"],
         parameter_dictionary["equilibrium"]["initial_distribution"]["Collector"]
         ])

    if setup == "decentralised":
        P = odeint(get_change, P0, t)

    elif setup == "centralised":
        P = odeint(get_change_centralised, P0, t)

    return P,t


def get_many_final_distributions(parameter_file, payoff_local, setup):
    parameter_dictionary = json.loads(open(parameter_file).read())
    total_time = parameter_dictionary["total_time"]
    intervals = parameter_dictionary["intervals"]
    num_samples = parameter_dictionary["num_samples"]

    global payoff
    payoff = payoff_local

    t = np.linspace(0, intervals, total_time)
    initial_distributions = sample_distributions(num_samples)
    final_distributions = [None] * num_samples

    for index in range(len(initial_distributions)):
        P0 = initial_distributions[index]

        if setup == "decentralised":
            dist = odeint(get_change, P0, t)[-1]

        elif setup == "centralised":
            dist = odeint(get_change_centralised, P0, t)[-1]

        for i in range(len(dist)):
            dist[i] = round(dist[i], 4)

        final_distributions[index] = dist

    return final_distributions


def get_team_payoff(payoff, P):
    """
    Given a payoff matrix and distribution of strategies, get the payoff of a team.
    Payoff is negated for the purpose of optimisation.
    Returns infinity for invalid distributions (also for purpose of optimisation)

    @param payoff: Payoff matrix
    @param P: Distribution of strategies
    @return: Infinity for invalid distribution. Negated average team payoff otherwise
    """
    #assert sum(P) == 1.0, "Population distribution does not sum to 1"
    if sum(P) > 1.0:
        return float('inf')

    f_novice = sum([P[i] * (payoff["Novice"][strategy[i]] + payoff[strategy[i]]["Novice"]) for i in range(len(strategy))])
    f_generalist = sum([P[i] * (payoff["Generalist"][strategy[i]] + payoff[strategy[i]]["Generalist"]) for i in range(len(strategy))])
    f_dropper = sum([P[i] * (payoff["Dropper"][strategy[i]] + payoff[strategy[i]]["Dropper"]) for i in range(len(strategy))])
    f_collector = sum([P[i] * (payoff["Collector"][strategy[i]] + payoff[strategy[i]]["Collector"]) for i in range(len(strategy))])
    f_avg = P[0] * f_novice + P[1] * f_generalist + P[2] * f_dropper + P[3] * f_collector

    return -1. * f_avg


def optimise_payoff(payoff):
    """
    Uses differential evolution to find the optimal team distribution, given a particular payoff matrix

    @param payoff: Payoff matrix
    @return: A tuple where the first item is a comma-separated string of the percentages of each strategy in the optimal
    distribution, and the second item is the payoff of that optimal team
    """
    find_payoff = partial(get_team_payoff, payoff)
    bounds = [(0, 1), (0, 1), (0, 1), (0, 1)]
    res = differential_evolution(find_payoff, bounds)

    return ", ".join(["{:.2f}".format(x) for x in res.x]), str(-1.*res.fun)


def calculate_price_of_anarchy(parameter_file, all_distributions):
    parameter_dictionary = json.loads(open(parameter_file).read())
    assert parameter_dictionary["experiment_type"] == "price_of_anarchy", "Experiment type is not price_of_anarchy"

    payoff_matrices = get_payoff_matrix(parameter_file)

    slope_list = parameter_dictionary["price_of_anarchy"]["slopes"]
    price_list = []

    for payoff in payoff_matrices:
        optimal_distribution, optimal_team_payoff = optimise_payoff(payoff)
        optimal_team_payoff = float(optimal_team_payoff)
        find_payoff = partial(get_team_payoff, payoff)

        if not all_distributions:
            optimal_distribution = get_final_distribution(parameter_file, payoff, setup="centralised")[0][-1]
            optimal_team_payoff = -1. * get_team_payoff(payoff, optimal_distribution)

            selfish_distribution = get_final_distribution(parameter_file, payoff, setup="decentralised")[0][-1]
            selfish_payoff = -1. * get_team_payoff(payoff, selfish_distribution)

            price_of_anarchy = optimal_team_payoff / selfish_payoff
            price_list += [price_of_anarchy]

        else:
            #optimal_distributions = get_many_final_distributions(parameter_file, payoff, setup="centralised")
            selfish_distributions = get_many_final_distributions(parameter_file, payoff, setup="decentralised")

            #optimal_payoffs = -1. * np.array(list(map(find_payoff, optimal_distributions)))
            selfish_payoffs = -1. * np.array(list(map(find_payoff, selfish_distributions)))
            #price_of_anarchy = np.mean(optimal_payoffs / selfish_payoffs)
            price_of_anarchy = np.mean([optimal_team_payoff/selfish_payoff for selfish_payoff in selfish_payoffs])
            price_list += [price_of_anarchy]

    plt.plot(slope_list, price_list)
    plt.title("Price of Anarchy as Slope Increases")
    plt.ylabel("Price of Anarchy")
    plt.xlabel("Slope")
    path = "/".join([el for el in parameter_file.split("/")[:-1]]) + "/analysis"
    filename = parameter_file.split("/")[-1].strip(".json") + ".png"
    plt.savefig(os.path.join(path, filename))


def plot_results(P, t, parameter_file, setup):
    novice, generalist, dropper, collector = P.T
    plt.plot(t, novice, label='novice')
    plt.plot(t, generalist, label='generalist')
    plt.plot(t, dropper, label='dropper')
    plt.plot(t, collector, label='collector')
    plt.grid()
    plt.ylim(-0.1, 1.1)
    plt.title("Distribution of Strategies Over Time")
    plt.ylabel("Proportion of population")
    plt.xlabel("Time")
    plt.legend(loc='best')
    path = "/".join([el for el in parameter_file.split("/")[:-1]]) + "/analysis"
    filename = parameter_file.split("/")[-1].strip(".json") + f"_{setup}.png"
    plt.savefig(os.path.join(path, filename))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot evolutionary dynamics')
    parser.add_argument('--parameters', action="store", dest="parameters")
    parser.add_argument('--setup', action="store", dest="setup")
    parser.add_argument('--all_distributions', action="store", dest="all_distributions")
    parameter_file = parser.parse_args().parameters
    setup = parser.parse_args().setup
    all_distributions = parser.parse_args().all_distributions

    if not all_distributions or all_distributions != "True":
        all_distributions = False
    else:
        all_distributions = True

    parameter_dictionary = json.loads(open(parameter_file).read())

    if parameter_dictionary["experiment_type"] == "equilibrium":
        payoff = get_payoff_matrix(parameter_file)
        if not all_distributions:
            P, t = get_final_distribution(parameter_file, payoff, setup)
            plot_results(P, t, parameter_file, setup)
        else:
            raise RuntimeError("Cannot generate equilibrium plot for all distributions")

    elif parameter_dictionary["experiment_type"] == "price_of_anarchy":
        calculate_price_of_anarchy(parameter_file, all_distributions)

    #print(parameter_file.split("/")[-1].strip(".json"))
    #print(optimise_payoff(payoff))
    #print("\n")


