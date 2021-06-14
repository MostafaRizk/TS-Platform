from learning.learner_parent import Learner


class FullyCentralisedLearner(Learner):
    def __init__(self, calculator):
        super().__init__(calculator)

    def learn(self, logging):
        pass

    # Helpers ---------------------------------------------------------------------------------------------------------

    def generate_model_name(self, fitness):
        pass