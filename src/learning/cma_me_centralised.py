import cma
import copy
import sys
import os
import numpy as np
import ray

from fitness import FitnessCalculator
from learning.learner_centralised import CentralisedLearner
from learning.cma_parent import CMALearner
from glob import glob
from io import StringIO
from learning.rwg import RWGLearner
from learning.cma_centralised import CentralisedCMALearner
from helpers.novelty_map import NoveltyMap
from helpers.elite import Elite
from learning.learner_parent import Learner

@ray.remote
def learn_in_parallel(fitness_calculator, agent_pop, spec_flag):
    return fitness_calculator.calculate_fitness_of_agent_population(agent_pop, spec_flag)


class CentralisedCMAMELearner(CentralisedCMALearner):
    def __init__(self, calculator):
        super().__init__(calculator)
        shape_tuple = (8, 8)  # TODO: Don't hard-code
        seed = self.parameter_dictionary['general']['seed']
        self.novelty_map = NoveltyMap(shape_tuple, seed)

    def learn(self, logging=True):
        # Put CMA output in a buffer for logging to a file at the end of the function
        old_stdout = sys.stdout
        sys.stdout = mystdout = StringIO()

        options = {'seed': self.parameter_dictionary['general']['seed'],
                   'maxiter': self.parameter_dictionary['algorithm']['cma-me']['generations'],
                   'popsize': self.get_genome_population_length(),
                   'tolx': self.parameter_dictionary['algorithm']['cma-me']['tolx'],
                   'tolfunhist': self.parameter_dictionary['algorithm']['cma-me']['tolfunhist'],
                   'tolflatfitness': self.parameter_dictionary['algorithm']['cma-me']['tolflatfitness'],
                   'tolfun': self.parameter_dictionary['algorithm']['cma-me']['tolfun']}

        # Initialise cma with a mean genome and sigma
        seed_genome, seed_fitness = self.get_seed_genome()
        es = cma.CMAEvolutionStrategy(seed_genome, self.parameter_dictionary['algorithm']['cma-me']['sigma'], options)
        generation = es.result.iterations
        best_genome, best_fitness = seed_genome, seed_fitness
        num_threads = self.num_agents

        parents = []
        parent_scores = []

        if self.parameter_dictionary["algorithm"]["agent_population_size"] % num_threads != 0:
            raise RuntimeError("Agent population is not divisible by the number of parallel threads")

        # Learning loop
        while not es.stop():
            # Get population of genomes to be used this generation
            genome_population = es.ask()
            print("Asked")

            # For homogeneous teams rewarding at the individual level,
            if self.team_type == "homogeneous" and self.reward_level == "individual":
                """
                                new_population = []

                                for ind in genome_population:
                                    ind_1 = copy.deepcopy(ind)
                                    ind_2 = copy.deepcopy(ind)
                                    new_population += [ind_1, ind_2]

                                genome_population = new_population
                """
                raise RuntimeError("This configuration is not supported yet")

            # Convert genomes to agents
            agent_population = self.convert_genomes_to_agents(genome_population)
            agent_pop_size = len(agent_population)
            agent_fitness_lists = []

            if self.multithreading:
                remainder_agents = agent_pop_size % (self.num_agents ** 2)
                divisible_pop_size = agent_pop_size - remainder_agents
                parallel_threads = []

                for i in range(0, num_threads + 1):
                    start = i * (divisible_pop_size // num_threads)
                    end = (i + 1) * (divisible_pop_size // num_threads)
                    mini_pop = agent_population[start:end]

                    # Makes sure that each core gets a population that can be divided into teams of size self.num_agents
                    if i == num_threads and remainder_agents != 0:
                        mini_pop += agent_population[end:end + remainder_agents]

                    parallel_threads += [
                        learn_in_parallel.remote(self.fitness_calculator, mini_pop, self.calculate_specialisation)]

                parallel_results = ray.get(parallel_threads)
                # team_specialisations = []

                for element in parallel_results:
                    agent_fitness_lists += element[0]
                    # team_specialisations += element[1]

            else:
                agent_fitness_lists, team_specialisations, behaviour_characterisations = self.fitness_calculator.calculate_fitness_of_agent_population(agent_population, self.calculate_specialisation)

            # Convert agent fitnesses into genome fitnesses
            genome_fitness_lists = self.get_genome_fitnesses_from_agent_fitnesses(agent_fitness_lists)
            genome_fitness_average = [np.mean(fitness_list) for fitness_list in genome_fitness_lists]
            behaviour_characterisation_average = [np.mean(bc, axis=0) for bc in behaviour_characterisations]

            # Updating distribution according to CMA-ME Improvement Emitter
            for index,genome in enumerate(genome_population):
                bc = behaviour_characterisation_average[index]
                fitness = genome_fitness_average[index]

                if fitness > best_fitness:
                    best_genome = genome
                    best_fitness = fitness

                elite = self.novelty_map.get_elite(bc)

                if not elite:
                    change_in_fitness = fitness
                    parents += [genome]
                    parent_scores += [change_in_fitness]
                    elite = Elite(genome, fitness, bc)
                    self.novelty_map.set_elite(elite)

                elif elite.fitness < fitness:
                    change_in_fitness = fitness - elite.fitness
                    parents += [genome]
                    parent_scores += [change_in_fitness]
                    elite = Elite(genome, fitness, bc)
                    self.novelty_map.set_elite(elite)

            if parents and len(parents) >= es.popsize:
                # Update the algorithm with the new fitness evaluations
                # CMA minimises fitness so we negate the fitness values
                es.tell(parents, [-f for f in parent_scores])
                generation += 1
                #print("Told")
                parents = []
                parent_scores = []

            elif not parents:
                elite = self.novelty_map.get_random_elite()
                options['maxiter'] = self.parameter_dictionary['algorithm']['cma-me']['generations'] - generation
                es = cma.CMAEvolutionStrategy(elite.genome, self.parameter_dictionary['algorithm']['cma-me']['sigma'], options)

            #print(generation)

            if generation % self.logging_rate == 0:
                self.log(best_genome, best_fitness, generation, seed_fitness)

            es.disp()

        print(f"Best fitness is {best_fitness}")

        print(es.stop())

        # Save best model
        model_name = self.generate_model_name(best_fitness)
        self.log(best_genome, best_fitness, "final", seed_fitness)

        # Log evolution details to file
        log_file_name = model_name + ".log"
        f = open(log_file_name, "w")
        f.write(mystdout.getvalue())
        f.close()

        # Reset output stream
        sys.stdout = old_stdout

        return best_genome, best_fitness

    def generate_model_name(self, fitness):
        return CentralisedCMAMELearner.get_model_name_from_dictionary(self.parameter_dictionary, fitness)

