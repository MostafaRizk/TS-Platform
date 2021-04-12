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

strategy = ["Novice", "Generalist", "Dropper", "Collector"]


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
    Gets relevant values from json file and prepares the payoff matrix
    @param parameter_file: Path to json file
    @return: Payoff matrix for each pair of strategies
    """

    parameter_dictionary = json.loads(open(parameter_file).read())

    e = parameter_dictionary["e"]
    g = parameter_dictionary["g"]
    b = parameter_dictionary["b"]
    f = parameter_dictionary["f"]
    c = parameter_dictionary["c"]

    s = parameter_dictionary["s"]  # Sliding speed

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

    return payoff


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


def get_final_distribution(parameter_file, payoff, setup):
    parameter_dictionary = json.loads(open(parameter_file).read())

    P0 = np.array([parameter_dictionary["initial_distribution"]["Novice"],
         parameter_dictionary["initial_distribution"]["Generalist"],
         parameter_dictionary["initial_distribution"]["Dropper"],
         parameter_dictionary["initial_distribution"]["Collector"]
         ])

    t = np.linspace(0, 50, 1000)

    if setup == "decentralised":
        P = odeint(get_change, P0, t)
    elif setup == "centralised":
        P = odeint(get_change_centralised, P0, t)

    return P,t


def get_team_payoff(payoff, P):
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
    find_payoff = partial(get_team_payoff, payoff)
    bounds = [(0, 1), (0, 1), (0, 1), (0, 1)]
    res = differential_evolution(find_payoff, bounds)

    return ", ".join(["{:.2f}".format(x) for x in res.x]), str(-1.*res.fun)


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
    parameter_file = parser.parse_args().parameters
    setup = parser.parse_args().setup

    payoff = get_payoff_matrix(parameter_file)
    #P,t = get_final_distribution(parameter_file, payoff, setup)
    #plot_results(P, t, parameter_file, setup)

    print(parameter_file.split("/")[-1].strip(".json"))
    print(optimise_payoff(payoff))
    print("\n")
