import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from pylab import legend

def plot_fitness_population(results_file, graph_file):
    # Read data
    data = pd.read_csv(results_file)
    #trimmed_data = data[[" Team Type", " Slope Angle", " Fitness"]]
    trimmed_data = data[[" Team Type", " Slope Angle", " Arena Length", " Sigma", " Population", " Fitness"]]

    # Format data
    results = {"20": {"homogeneous": [], "heterogeneous": []},
               "40": {"homogeneous": [], "heterogeneous": []},
               "60": {"homogeneous": [], "heterogeneous": []},
               "80": {"homogeneous": [], "heterogeneous": []},
               "100": {"homogeneous": [], "heterogeneous": []}}

    data_points = 0

    for index, row in trimmed_data.iterrows():
        if row[" Arena Length"] == 8 and row[" Sigma"] == 0.05 and row[" Slope Angle"] == 40:
            results[str(row[" Population"])][row[" Team Type"]] += [row[" Fitness"]*-1]
            data_points += 1

    print(f"Quality check: There were {data_points} data points out of {len(trimmed_data)}. ")

    for key in results.keys():
        print(len(results[key]["homogeneous"]))
        print(len(results[key]["heterogeneous"]))

    # Plot data
    fig1, ax1 = plt.subplots()
    ax1.set_title('Best Fitness Score vs Population Size')
    ax1.set_ylim(0, 50)
    ax1.set_ylabel('Fitness')
    ax1.set_xlabel('Population')

    positions = [1, 4, 7, 10, 13]
    pop_sizes = [20, 40, 60, 80, 100]

    for i in range(len(pop_sizes)):
        pop_size = pop_sizes[i]
        box1 = ax1.boxplot([results[str(pop_size)]["homogeneous"]], positions=[positions[i]], widths=0.3)
        box2 = ax1.boxplot([results[str(pop_size)]["heterogeneous"]], positions=[positions[i]+1], widths=0.3)
        for item in ['boxes', 'whiskers', 'fliers', 'medians', 'caps']:
            plt.setp(box1[item], color="red")
            plt.setp(box2[item], color="blue")

    ax1.set_xticklabels(pop_sizes)
    ax1.set_xticks([x + 0.5 for x in positions])

    hB, = ax1.plot([1, 1], 'r-')
    hR, = ax1.plot([1, 1], 'b-')
    legend((hB, hR), ('Homogeneous', 'Heterogeneous'))
    hB.set_visible(False)
    hR.set_visible(False)

    #plt.show()
    plt.savefig(graph_file)

def plot_fitness_sigma(results_file, graph_file):
    # Read data
    data = pd.read_csv(results_file)
    #trimmed_data = data[[" Team Type", " Slope Angle", " Fitness"]]
    trimmed_data = data[[" Team Type", " Slope Angle", " Arena Length", " Sigma", " Population", " Fitness"]]

    # Format data
    results = {"0.05": {"homogeneous": [], "heterogeneous": []},
               "0.1": {"homogeneous": [], "heterogeneous": []},
               "0.2": {"homogeneous": [], "heterogeneous": []},
               "0.3": {"homogeneous": [], "heterogeneous": []},
               "0.4": {"homogeneous": [], "heterogeneous": []}}

    data_points = 0

    for index, row in trimmed_data.iterrows():
        if row[" Arena Length"] == 8 and row[" Population"] == 40 and row[" Slope Angle"] == 40:
            results[str(row[" Sigma"])][row[" Team Type"]] += [row[" Fitness"]*-1]
            data_points += 1

    print(f"Quality check: There were {data_points} data points out of {len(trimmed_data)}. ")

    for key in results.keys():
        print(len(results[key]["homogeneous"]))
        print(len(results[key]["heterogeneous"]))

    # Plot data
    fig1, ax1 = plt.subplots()
    ax1.set_title('Best Fitness Score vs Sigma')
    ax1.set_ylim(0, 50)
    ax1.set_ylabel('Fitness')
    ax1.set_xlabel('Sigma')

    positions = [1, 4, 7, 10, 13]
    sigmas = [0.05, 0.1, 0.2, 0.3, 0.4]

    for i in range(len(sigmas)):
        sigma = sigmas[i]
        box1 = ax1.boxplot([results[str(sigma)]["homogeneous"]], positions=[positions[i]], widths=0.3)
        box2 = ax1.boxplot([results[str(sigma)]["heterogeneous"]], positions=[positions[i]+1], widths=0.3)
        for item in ['boxes', 'whiskers', 'fliers', 'medians', 'caps']:
            plt.setp(box1[item], color="red")
            plt.setp(box2[item], color="blue")

    ax1.set_xticklabels(sigmas)
    ax1.set_xticks([x + 0.5 for x in positions])

    hB, = ax1.plot([1, 1], 'r-')
    hR, = ax1.plot([1, 1], 'b-')
    legend((hB, hR), ('Homogeneous', 'Heterogeneous'))
    hB.set_visible(False)
    hR.set_visible(False)

    #plt.show()
    plt.savefig(graph_file)

def plot_fitness_slope_length(results_file, graph_file):
    # Read data
    data = pd.read_csv(results_file)
    #trimmed_data = data[[" Team Type", " Slope Angle", " Fitness"]]
    trimmed_data = data[[" Team Type", " Slope Angle", " Arena Length", " Sigma", " Population", " Fitness"]]

    # Format data
    results = {"8": {"homogeneous": [], "heterogeneous": []},
               "12": {"homogeneous": [], "heterogeneous": []},
               "16": {"homogeneous": [], "heterogeneous": []},
               "20": {"homogeneous": [], "heterogeneous": []}}

    data_points = 0

    for index, row in trimmed_data.iterrows():
        if row[" Sigma"] == 0.05 and row[" Population"] == 40 and row[" Slope Angle"] == 40:
            results[str(row[" Arena Length"])][row[" Team Type"]] += [row[" Fitness"]*-1]
            data_points += 1

    print(f"Quality check: There were {data_points} data points out of {len(trimmed_data)}. ")

    for key in results.keys():
        print(len(results[key]["homogeneous"]))
        print(len(results[key]["heterogeneous"]))

    # Plot data
    fig1, ax1 = plt.subplots()
    ax1.set_title('Best Fitness Score vs Slope Length')
    ax1.set_ylim(0, 50)
    ax1.set_ylabel('Fitness')
    ax1.set_xlabel('Slope Length')

    positions = [1, 4, 7, 10]
    lengths = [8, 12, 16, 20]

    for i in range(len(lengths)):
        length = lengths[i]
        box1 = ax1.boxplot([results[str(length)]["homogeneous"]], positions=[positions[i]], widths=0.3)
        box2 = ax1.boxplot([results[str(length)]["heterogeneous"]], positions=[positions[i]+1], widths=0.3)
        for item in ['boxes', 'whiskers', 'fliers', 'medians', 'caps']:
            plt.setp(box1[item], color="red")
            plt.setp(box2[item], color="blue")

    ax1.set_xticklabels([x-4 for x in lengths])
    ax1.set_xticks([x + 0.5 for x in positions])

    hB, = ax1.plot([1, 1], 'r-')
    hR, = ax1.plot([1, 1], 'b-')
    legend((hB, hR), ('Homogeneous', 'Heterogeneous'))
    hB.set_visible(False)
    hR.set_visible(False)

    #plt.show()
    plt.savefig(graph_file)

def plot_fitness_sliding_speed(results_file, graph_file):
    # Read data
    data = pd.read_csv(results_file)
    #trimmed_data = data[[" Team Type", " Slope Angle", " Fitness"]]
    trimmed_data = data[[" Team Type", " Slope Angle", " Arena Length", " Sigma", " Population", " Fitness"]]

    # Format data
    results = {"40": {"homogeneous": [], "heterogeneous": []},
               "100": {"homogeneous": [], "heterogeneous": []},
               "120": {"homogeneous": [], "heterogeneous": []},
               "140": {"homogeneous": [], "heterogeneous": []}}

    data_points = 0

    for index, row in trimmed_data.iterrows():
        if row[" Sigma"] == 0.05 and row[" Population"] == 40 and row[" Arena Length"] == 20:
            results[str(row[" Slope Angle"])][row[" Team Type"]] += [row[" Fitness"]*-1]
            data_points += 1

    print(f"Quality check: There were {data_points} data points out of {len(trimmed_data)}. ")

    for key in results.keys():
        print(len(results[key]["homogeneous"]))
        print(len(results[key]["heterogeneous"]))

    # Plot data
    fig1, ax1 = plt.subplots()
    ax1.set_title('Best Fitness Score vs Sliding Speed')
    ax1.set_ylim(0, 50)
    ax1.set_ylabel('Fitness')
    ax1.set_xlabel('Sliding Speed')

    positions = [1, 4, 7, 10]
    speeds = [40, 100, 120, 140]

    for i in range(len(speeds)):
        speed = speeds[i]
        box1 = ax1.boxplot([results[str(speed)]["homogeneous"]], positions=[positions[i]], widths=0.3)
        box2 = ax1.boxplot([results[str(speed)]["heterogeneous"]], positions=[positions[i]+1], widths=0.3)
        for item in ['boxes', 'whiskers', 'fliers', 'medians', 'caps']:
            plt.setp(box1[item], color="red")
            plt.setp(box2[item], color="blue")

    ax1.set_xticklabels([int(x/10) for x in speeds])
    ax1.set_xticks([x + 0.5 for x in positions])

    hB, = ax1.plot([1, 1], 'r-')
    hR, = ax1.plot([1, 1], 'b-')
    legend((hB, hR), ('Homogeneous', 'Heterogeneous'))
    hB.set_visible(False)
    hR.set_visible(False)

    #plt.show()
    plt.savefig(graph_file)


plot_fitness_population(f"results.csv", f"fitness_population.png")
plot_fitness_sigma(f"results.csv", f"fitness_sigma.png")
plot_fitness_slope_length(f"results.csv", f"fitness_slope_length.png")
plot_fitness_sliding_speed(f"results.csv", f"fitness_sliding_speed.png")


