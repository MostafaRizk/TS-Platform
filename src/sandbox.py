


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
        slope_list = [parameter_dictionary["equilibrium"]["s"]]  # Sliding speed

    elif parameter_dictionary["experiment_type"] == "price_of_anarchy":
        slope_list = parameter_dictionary["price_of_anarchy"]["slopes"]

    payoff_matrices = []

    for s in slope_list:
        # Payoff for each strategy when it is paired with another strategy (when there is a slope)
        payoff = {"Novice": {"Novice": s * theta(s) * f + (1 - theta(s)) * b,
                             "Generalist": s * theta(s) * f + (1 - theta(s)) * b,
                             "Dropper": s * theta(s) * f + (1 - theta(s)) * b,
                             "Collector": s * theta(s) * f + (1 - theta(s)) * b},

                  "Generalist": {"Novice": g - s * c,
                                 "Generalist": g - s * c,
                                 "Dropper": g - s * c,
                                 "Collector": g - s * c},

                  "Dropper": {"Novice": s * theta(s) * f + (1 - theta(s)) * b,
                              "Generalist": s * theta(s) * f + (1 - theta(s)) * b,
                              "Dropper": s * theta(s) * f + (1 - theta(s)) * b,
                              "Collector": s * theta(s) * e + (1 - theta(s)) * b},

                  "Collector": {"Novice": b,
                                "Generalist": b,
                                "Dropper": s * theta(s) * e + (1 - theta(s)) * b,
                                "Collector": b}}

        payoff_matrices += [payoff]

    if len(payoff_matrices) == 1:
        return payoff_matrices[0]
    else:
        return payoff_matrices





def get_change_centralised(P, t=0):
    f_novice = sum(
        [P[i] * (payoff["Novice"][strategy[i]] + payoff[strategy[i]]["Novice"]) for i in range(len(strategy))])
    f_generalist = sum(
        [P[i] * (payoff["Generalist"][strategy[i]] + payoff[strategy[i]]["Generalist"]) for i in range(len(strategy))])
    f_dropper = sum(
        [P[i] * (payoff["Dropper"][strategy[i]] + payoff[strategy[i]]["Dropper"]) for i in range(len(strategy))])
    f_collector = sum(
        [P[i] * (payoff["Collector"][strategy[i]] + payoff[strategy[i]]["Collector"]) for i in range(len(strategy))])
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

    # total_time = parameter_dictionary["equilibrium"]["total_time"]

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

    return P, t











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

    # print(parameter_file.split("/")[-1].strip(".json"))
    # print(optimise_payoff(payoff))
    # print("\n")


