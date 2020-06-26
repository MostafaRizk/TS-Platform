
def genetic_algorithm(calculator, seed_value, output_selection_method, num_generations=2000, population_size=100,
                      num_trials=3, mutation_rate=0.01,
                      elitism_percentage=0.05):
    # population_size = 10
    # num_generations = 40
    # num_trials = 3
    # mutation_rate = 0.01
    crossover_rate = 0.3  # Single-point crossover
    # elitism_percentage = 0.05  # Generational replacement with roulette-wheel selection
    num_elite = max(2, int(population_size * elitism_percentage))

    model_saving_rate = 10

    # Create randomly initialised population
    population = []

    for i in range(population_size):
        individual = TinyAgent(calculator.get_observation_size(), calculator.get_action_size(), output_selection_method,
                               seed_value)
        individual.load_weights()  # No parameters means random weights are generated
        population += [individual]
        seed_value += 1

    fitness_scores = [-math.inf] * population_size
    best_individual = None

    for generation in range(num_generations):

        for i in range(population_size):
            # Get fitness of individual and add to fitness_scores
            avg_fitness = 0
            for trial in range(num_trials):
                avg_fitness += calculator.calculate_fitness(population[i])
            avg_fitness /= num_trials
            fitness_scores[i] = avg_fitness

        new_population = []

        # Add elite to new_population
        for e in range(num_elite):
            best_score = -math.inf
            best_index = 0

            # Find best individual (excluding last selected elite)
            for ind in range(population_size):
                if fitness_scores[ind] > best_score:
                    best_score = fitness_scores[ind]
                    best_index = ind

            # Add best individual to new population and exclude them from next iteration
            new_population += [copy.deepcopy(population[best_index])]
            fitness_scores[best_index] = -math.inf

        best_individual = new_population[0]

        if generation % model_saving_rate == 0:
            print(f"Best score at generation {generation} is {best_score} ")
            best_individual.save_model(f"{output_selection_method}_{generation}")

        # Create new generation
        for i in range(population_size - num_elite):
            # Choose parents
            parent1_genome = new_population[0].get_weights()
            parent2_genome = new_population[1].get_weights()

            # Do crossover
            rng = calculator.get_rng()
            crossover_point = rng.randint(low=0, high=len(parent1_genome))
            child_genome = np.append(parent1_genome[0:crossover_point], parent2_genome[crossover_point:])

            # Mutate with probability and add to population
            # For every gene in the genome (i.e. weight in the neural network)
            for j in range(len(child_genome)):
                decimal_places = 16
                gene = child_genome[j]

                # Create a bit string from the float of the genome
                bit_string = bin(int(gene * (10 ** decimal_places)))  # Gene is a float, turn into int
                start_index = bit_string.find('b') + 1  # Prefix is either 0b or -0b. Start after 'b'
                bit_list = list(bit_string[start_index:])

                # Iterate through bit string and flip each bit according to mutation probability
                for b in range(len(bit_list)):
                    random_number = rng.rand()

                    if random_number < mutation_rate:
                        if bit_list[b] == '0':
                            bit_list[b] = '1'
                        else:
                            bit_list[b] = '0'

                mutated_gene = float(int(''.join(bit_list), 2)) / (10 ** decimal_places)
                child_genome[j] = mutated_gene

            # Create child individual and add to population
            child = TinyAgent(calculator.get_observation_size(), calculator.get_action_size(), output_selection_method,
                              seed_value)
            child.load_weights(child_genome)
            new_population += [child]

        # Reset values for population and fitness scores
        population = copy.deepcopy(new_population)
        fitness_scores = [-math.inf] * population_size

    return best_individual


def grammatical_evolution(random_seed, num_generations=2000, population_size=100,
                          num_trials=3, mutation_rate=0.01,
                          elitism_percentage=0.05):
    # Run PonyGE
    # python3 ponyge.py --parameters task_specialisation.txt --random_seed random_seed
    pass


def q_learning(calculator, num_episodes, random_seed, batch_size):
    # Information for BasicQ
    front = [[0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0]]
    locations = [[0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0]]
    has_obj = [[0], [1]]

    possible_observations = []

    for x in front:
        for y in locations:
            for z in has_obj:
                possible_observations += [np.array([x + y + z])]

    # agent = BasicQAgent(possible_observations, calculator.get_action_size(), random_seed, batch_size=batch_size)

    agent = DQNAgent(calculator.get_observation_size(), calculator.get_action_size(), random_seed,
                     batch_size=128)  # calculator.simulation_length*calculator.env.num_robots)

    for e in range(num_episodes):
        render = False
        # if e%10 == 0:
        #    render = True
        score, agent = calculator.calculate_fitness(agent, num_trials=5, render=render, learning_method="DQN")
        print(f'Score at episode {e} is {score}')

        if e % 100 == 99:
            agent.save(f"Q_best_{e}")
            agent.display()

    return agent

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

