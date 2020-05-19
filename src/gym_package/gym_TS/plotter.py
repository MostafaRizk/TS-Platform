import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from copy import deepcopy

from pylab import legend


def plot_hardcoded_fitness(results_file, graph_file):
    data = pd.read_csv(results_file)
    strategies = ["Generalist-Generalist", "Generalist-Dropper", "Generalist-Collector", "Dropper-Dropper", "Dropper-Collector", "Collector-Collector"]
    missing_strategies = [strategies[:] for x in range(50)]

    results = {"Generalist-Generalist": {"Individual 1": [], "Individual 2": []},
               "Generalist-Dropper": {"Individual 1": [], "Individual 2": []},
               "Generalist-Collector": {"Individual 1": [], "Individual 2": []},
               "Dropper-Dropper": {"Individual 1": [], "Individual 2": []},
               "Dropper-Collector": {"Individual 1": [], "Individual 2": []},
               "Collector-Collector": {"Individual 1": [], "Individual 2": []}}

    data_points = 0

    for index, row in data.iterrows():
        results[str(row["Strategy"])]["Individual 1"] += [row["Individual 1 Score"]]
        results[str(row["Strategy"])]["Individual 2"] += [row["Individual 2 Score"]]
        data_points += 1
        print(index)
        missing_strategies[int(row["Seed"])-1].remove(row["Strategy"])

    print(f"Quality check: There were {data_points} data points out of {len(data)}. ")
    for i in range(len(missing_strategies)):
        if missing_strategies[i]:
            seed = i+1
            print(f"Seed {seed}: {missing_strategies[i]}")

    # Plot data
    fig1, ax1 = plt.subplots(figsize=(12,4))
    ax1.set_title('Fitness Scores of Hardcoded Strategies')
    ax1.set_ylim(-1000, 100000)
    ax1.set_ylabel('Fitness')
    ax1.set_xlabel('Team Strategy')

    positions = [1, 4, 7, 10, 13, 16]

    for i in range(len(strategies)):
        strategy = strategies[i]
        box1 = ax1.boxplot(results[strategy]["Individual 1"], positions=[positions[i]], widths=0.3)
        box2 = ax1.boxplot(results[strategy]["Individual 2"], positions=[positions[i]+1], widths=0.3)
        for item in ['boxes', 'whiskers', 'fliers', 'medians', 'caps']:
            plt.setp(box1[item], color="red")
            plt.setp(box2[item], color="blue")

    ax1.set_xticklabels(strategies)
    ax1.set_xticks([x + 0.5 for x in positions])

    hB, = ax1.plot([1, 1], 'r-')
    hR, = ax1.plot([1, 1], 'b-')
    legend((hB, hR), ('Individual 1', 'Individual 2'))
    hB.set_visible(False)
    hR.set_visible(False)

    #plt.savefig(graph_file)
    plt.show()

plot_hardcoded_fitness("hardcoded_results.csv", "hardcoded_fitness.png")