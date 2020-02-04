import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def plot_mean_fitness(results_file, graph_file):
    # Read data
    data = pd.read_csv(results_file)
    trimmed_data = data[[" Team Type", " Slope Angle", " Fitness"]]

    # Format data
    results = {"0": {"homogeneous": [], "heterogeneous": []},
               "10": {"homogeneous": [], "heterogeneous": []},
               "20": {"homogeneous": [], "heterogeneous": []},
               "30": {"homogeneous": [], "heterogeneous": []},
               "40": {"homogeneous": [], "heterogeneous": []}}

    for index, row in trimmed_data.iterrows():
        results[str(row[" Slope Angle"])][row[" Team Type"]] += [row[" Fitness"]*-1]

    # Plot data
    fig1, ax1 = plt.subplots()
    ax1.set_title('Mean Fitness')
    ax1.set_ylim(0, 30)
    ax1.boxplot([results["0"]["homogeneous"], results["0"]["heterogeneous"]], positions=[1, 2], widths=0.3)
    ax1.boxplot([results["40"]["homogeneous"], results["40"]["heterogeneous"]], positions=[4, 5], widths=0.3)
    ax1.set_xticklabels(['0°', '40°'])
    ax1.set_xticks([1.5, 4.5])
    #plt.show()
    plt.savefig(graph_file)


for i in range(20, 280, 20):
    plot_mean_fitness(f"{i}_results.csv", f"prelim_{i}.png")
