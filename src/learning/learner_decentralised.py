import copy

from learning.learner_parent import Learner


class DecentralisedLearner(Learner):
    def __init__(self, calculator):
        super().__init__(calculator)

        # A list of the 'best' representative from each learning population
        # (Definition of 'best' varies)
        self.representative_genomes = []

    def learn(self, logging):
        pass

    # Helpers ---------------------------------------------------------------------------------------------------------

    def insert_representative_genomes_in_population(self, genome_population, index):
        """
        Takes the best genomes from the populations of all the other learners and insert them after every genome in
        this learner's population e.g.
        [genome_1, ... genome_n], [rep_genome_1, rep_genome_2, rep_genome_3] ->
        [genome_1, rep_genome_1, rep_genome_2, rep_genome_3, ... genome_n, rep_genome_1, rep_genome_2, rep_genome_3]
        When agents are made from this population and passed to the fitness calculator, each agent will be partnered
        with the best agents from the other learners

        @param genome_population: Population of genomes for this learner
        @param index: Integer index of the representative genome for this population (so it can be skipped)
        @return: new_genome_population: List of combined genomes
        """
        new_genome_population = []
        teammate_genomes = copy.deepcopy(self.representative_genomes)
        teammate_genomes.pop(index)

        for genome in genome_population:
            new_genome_population += [genome]

            for teammate in teammate_genomes:
                new_genome_population += [teammate]

        return new_genome_population

    def remove_representative_fitnesses(self, genome_fitness_lists):
        """
        Given a list of the fitness values of all agents on all teams, remove the fitnesses of representative agents

        @param genome_fitness_lists: List of fitnesses of all agents/genomes evaluated
        @return: List of fitness values for only the genomes/agents in this learner's population
        """

        trimmed_fitness_list = []
        agents_per_team = self.num_agents

        for i in range(0, len(genome_fitness_lists), agents_per_team):
            trimmed_fitness_list += genome_fitness_lists[i]

        return trimmed_fitness_list

    def generate_model_name(self, fitness, agent_index):
        pass


