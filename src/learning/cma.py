import cma
import copy
import sys
from learning.learner_parent import Learner
from glob import glob
from io import StringIO


class CMALearner(Learner):
    def __init__(self, calculator):
        super().__init__(calculator)

        if self.parameter_dictionary['general']['algorithm_selected'] != "cma":
            raise RuntimeError(f"Cannot run cma. Parameters request "
                               f"{self.parameter_dictionary['general']['algorithm_selected']}")

        # Log every x many generations
        self.logging_rate = 20

    def learn(self):
        options = {'seed': self.parameter_dictionary['general']['seed'],
                   'maxiter': self.parameter_dictionary['algorithm']['cma']['generations'],
                   'popsize': self.get_genome_population_length(),
                   'tolx': self.parameter_dictionary['algorithm']['cma']['tolx'],
                   'tolfunhist': self.parameter_dictionary['algorithm']['cma']['tolfunhist']}

        # Initialise cma with a mean genome and sigma
        seed_genome = self.get_seed_genome()
        es = cma.CMAEvolutionStrategy(seed_genome, self.parameter_dictionary['algorithm']['cma']['sigma'], options)

        # Put CMA output in a buffer for logging to a file at the end of the function
        old_stdout = sys.stdout
        sys.stdout = mystdout = StringIO()

        # Learning loop
        while not es.stop():
            # Get population of genomes to be used this generation
            genome_population = es.ask()

            # For homogeneous teams rewarding at the individual level,
            if self.team_type == "homogeneous" and self.reward_level == "individual":
                new_population = []

                for ind in genome_population:
                    ind_1 = copy.deepcopy(ind)
                    ind_2 = copy.deepcopy(ind)
                    new_population += [ind_1, ind_2]

                genome_population = new_population

            # Convert genomes to agents
            agent_population = self.convert_genomes_to_agents(genome_population)

            # Get fitnesses of agents
            agent_fitnesses = self.fitness_calculator.calculate_fitness_of_agent_population(agent_population)

            # Convert agent fitnesses into genome fitnesses
            genome_fitnesses = self.get_genome_fitnesses_from_agent_fitnesses(agent_fitnesses)

            # Update the algorithm with the new fitness evaluations
            # CMA minimises fitness so we negate the fitness values
            es.tell(genome_population, [-f for f in genome_fitnesses])

            generation = es.result.iterations

            if generation % self.logging_rate == 0:
                self.log(es.result[0], -es.result[1], generation)

            es.disp()

        print(f"Best score is {-es.result[1]}")
        sys.stdout = old_stdout

        # Return best genome and its fitness value
        best_genome = es.result[0]
        best_fitness = -es.result[-1]
        return best_genome, best_fitness

    def get_genome_population_length(self):
        """
        Calculates how many genomes should be in the CMA population based on the number of agents

        @return: Integer representing the length of the genome population
        """
        # If teams are heterogneous and rewards are individual, there must be one genome per agent
        if self.team_type == "heterogeneous" and self.reward_level == "individual":
            return self.parameter_dictionary['algorithm']['agent_population_size']

        # For most configurations, each genome is either copied onto two agents or split across two agents
        else:
            return self.parameter_dictionary['algorithm']['agent_population_size'] / 2

    def get_seed_genome(self):
        """
        Get the seed genome to be used as cma's mean. The appropriate seed genome is the one that has the same
        parameter values as the current experiment
        @return:
        """
        # Looks for seedfiles with the same parameters as the current experiment
        parameters_in_name = Learner.get_core_params_in_model_name(self.parameter_dictionary)
        parameters_in_name[0] = "rwg"
        seedfile_prefix = "_".join([str(param) for param in parameters_in_name])
        possible_seedfiles = glob(f'{seedfile_prefix}*')

        # Makes sure there is only one unambiguous seedfile
        if len(possible_seedfiles) == 0:
            raise RuntimeError('No valid seed files')
        elif len(possible_seedfiles) > 1:
            raise RuntimeError('Too many valid seed files')
        else:
            return self.Agent.load_model_from_file(possible_seedfiles[0])

    def log(self, genome, genome_fitness, generation):
        # Get experiment parameter prefix list
        # Generate results filename
        pass

    def generate_model_name(self, fitness):
        """
        Create a name string for a model generated using the given parameter file and fitness value

        @param fitness: Fitness of the model to be saved
        @return:
        """
        parameters_in_name = Learner.get_core_params_in_model_name(self.parameter_dictionary)
        parameters_in_name += CMALearner.get_additional_params_in_model_name(self.parameter_dictionary)

        # Get fitness
        parameters_in_name += [fitness]

        # Put all in a string and return
        return "_".join([str(param) for param in parameters_in_name])

    @staticmethod
    def get_additional_params_in_model_name(parameter_dictionary):
        parameters_in_name = []

        # Get algorithm params for relevant algorithm
        parameters_in_name += [parameter_dictionary['algorithm']['agent_population_size']]
        parameters_in_name += [parameter_dictionary['algorithm']['cma']['sigma']]
        parameters_in_name += [parameter_dictionary['algorithm']['cma']['generations']]
        parameters_in_name += [parameter_dictionary['algorithm']['cma']['tolx']]
        parameters_in_name += [parameter_dictionary['algorithm']['cma']['tolfunhist']]

        return parameters_in_name


def cma_es(fitness_calculator, seed_value, sigma, model_name, results_file_name, team_type, selection_level,
           num_generations, num_teams):
    while not es.stop():
        iteration_number = es.result.iterations

        if iteration_number == 0:
            seed_fitness = -es.result[1]

        if iteration_number % LOG_EVERY == 0:
            # Log results to results file
            results = model_name.replace("_", ",")
            results += f",{log_file_name}, {seed_fitness}, {-es.result[1]}\n"
            intermediate_results_file_name = f"results_{iteration_number}.csv"

            if not os.path.exists(intermediate_results_file_name):
                results_file = open(intermediate_results_file_name, 'a')
                results_file.write(
                    "Algorithm Name, Team Type, Selection Level, Simulation Length, Num Generations, Num Trials, "
                    "Random Seed, Num Robots, Num Resources, Sensor Range, Slope Angle, Arena Length, "
                    "Arena Width, Cache Start, Slope Start, Source Start, Sigma, Population, Log File, "
                    "Seed Fitness, Evolved Fitness\n")
            else:
                results_file = open(intermediate_results_file_name, 'a')

            results_file.write(results)
            results_file.close()

            # Log genome
            # Split the genome and save both halves separately for heterogeneous setup
            if team_type == "heterogeneous" and selection_level == "team":
                best_individual_1 = TinyAgent(fitness_calculator.get_observation_size(),
                                              fitness_calculator.get_action_size(),
                                              seed=seed_value)
                best_individual_2 = TinyAgent(fitness_calculator.get_observation_size(),
                                              fitness_calculator.get_action_size(),
                                              seed=seed_value)

                # Split genome
                mid = int(len(es.result[0]) / 2)
                best_individual_1.load_weights(es.result[0][0:mid])
                best_individual_2.load_weights(es.result[0][mid:])

                best_individual_1.save_model(model_name + "_controller1_" + str(iteration_number) + "_")
                best_individual_2.save_model(model_name + "_controller2_" + str(iteration_number) + "_")

            else:
                best_individual = TinyAgent(fitness_calculator.get_observation_size(),
                                            fitness_calculator.get_action_size(),
                                            seed=seed_value)
                best_individual.load_weights(es.result[0])
                best_individual.save_model(model_name + "_" + str(iteration_number))

        es.disp()

    print(f"Best score is {-es.result[1]}")

    ''''''
    sys.stdout = old_stdout
    log_file.close()

    # Append results to results file. Create file if it doesn't exist
    results = model_name.replace("_", ",")
    results += f",{log_file_name}, {seed_fitness}, {-es.result[1]}\n"
    results_file = open(results_file_name, 'a')
    results_file.write(results)
    results_file.close()

    return es.result[0]
