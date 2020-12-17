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
    
    def insert_representative_genomes_in_population(self, genome_population, representative_genomes):
        pass

    def remove_representative_fitnesses(self, genome_fitness_lists):
        pass

