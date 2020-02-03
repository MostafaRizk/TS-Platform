import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def plot_mean_fitness(results_file):
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
    ax1.boxplot([results["20"]["homogeneous"], results["20"]["heterogeneous"]], positions=[4, 5], widths=0.3)
    ax1.set_xticklabels(['0°', '20°'])
    ax1.set_xticks([1.5, 4.5])
    #plt.show()
    plt.savefig("prelim_50.png")


plot_mean_fitness("50_results.csv")
