"""
Modelling slope foraging as a game and plotting the evolutionary dynamics

Strategies:
Novice-
Generalist-
Dropper-
Collector-
"""

import matplotlib.pyplot as plt

# Distribution of strategies over time (assumes everyone is a novice to begin with)
x = {"Novice": [0.05],
     "Generalist": [0.05],
     "Dropper": [0.45],
     "Collector": [0.45]}

e = 20
g = 5
b = -5
f = -10

dt = 0.1

# Payoff for each strategy when it is paired with another strategy (when there is a slope)
payoff = {"Novice": {"Novice": f,
                     "Generalist": g,
                     "Dropper": f,
                     "Collector": f},

          "Generalist": {"Novice": g,
                         "Generalist": g,
                         "Dropper": g,
                         "Collector": g},

          "Dropper": {"Novice": f,
                      "Generalist": f,
                      "Dropper": f,
                      "Collector": e},

          "Collector": {"Novice": b,
                        "Generalist": b,
                        "Dropper": e,
                        "Collector": b}}

# Average fitness of each strategy based on their proportions in the population
novice_fitness = [sum([x[strategy][0]*payoff["Novice"][strategy] for strategy in x])]
generalist_fitness = [sum([x[strategy][0]*payoff["Generalist"][strategy] for strategy in x])]
dropper_fitness = [sum([x[strategy][0]*payoff["Dropper"][strategy] for strategy in x])]
collector_fitness = [sum([x[strategy][0]*payoff["Collector"][strategy] for strategy in x])]

# Average fitness of population
average_fitness = x["Novice"][0]*novice_fitness[0] + x["Generalist"][0]*generalist_fitness[0] + \
                  x["Dropper"][0]*dropper_fitness[0] + x["Collector"][0]*collector_fitness[0]

novice_fitness[0] *= dt
generalist_fitness[0] *= dt
dropper_fitness[0] *= dt
collector_fitness[0] *= dt

for t in range(100):
    # Calculate fitness of each strategy and average fitness
    f_novice = sum([x[strategy][t]*payoff["Novice"][strategy] for strategy in x])
    f_generalist = sum([x[strategy][t]*payoff["Generalist"][strategy] for strategy in x])
    f_dropper = sum([x[strategy][t]*payoff["Dropper"][strategy] for strategy in x])
    f_collector = sum([x[strategy][t]*payoff["Collector"][strategy] for strategy in x])
    f_avg = x["Novice"][t]*f_novice + x["Generalist"][t]*f_generalist + x["Dropper"][t]*f_dropper + x["Collector"][t]*f_collector

    # Append to vectors
    novice_fitness += [f_novice * dt]
    generalist_fitness += [f_generalist * dt]
    dropper_fitness += [f_dropper * dt]
    collector_fitness += [f_collector * dt]

    # Differential equations to update proportions
    x["Novice"] += [x["Novice"][t] + (x["Novice"][t] * (f_novice - f_avg))*dt]
    x["Generalist"] += [x["Generalist"][t] + (x["Generalist"][t] * (f_generalist - f_avg)) * dt]
    x["Dropper"] += [x["Dropper"][t] + (x["Dropper"][t] * (f_dropper - f_avg)) * dt]
    x["Collector"] += [x["Collector"][t] + (x["Collector"][t] * (f_collector - f_avg)) * dt]

# Make plots
print(x)

plt.plot(x["Novice"], label='Novice')
plt.plot(x["Generalist"], label='Generalist')
plt.plot(x["Dropper"], label='Dropper')
plt.plot(x["Collector"], label='Collector')
plt.grid()
plt.ylim(0, 1)
plt.legend(loc='best')
plt.show()

