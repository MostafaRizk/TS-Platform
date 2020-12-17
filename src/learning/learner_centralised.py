from learning.learner_parent import Learner
from operator import add


class CentralisedLearner(Learner):
    def __init__(self, calculator):
        super().__init__(calculator)

    def learn(self, logging):
        pass

    # Helpers ---------------------------------------------------------------------------------------------------------

    def get_genome_fitnesses_from_agent_fitnesses(self, agent_fitness_lists):
        """
        Given a list of fitness lists of teams of agents, returns the fitness lists of the genomes they came from, based on the
        configuration of team type and reward level

        @param agent_fitnesses: A list of fitness lists for each agent in the agent population
        @return: A list of fitness lists for each genome that the agents came from
        """
        genome_fitness_lists = []

        if self.reward_level == "team":
            for i in range(0, len(agent_fitness_lists) - 1, self.num_agents):
                team_fitness_list = [0] * len(agent_fitness_lists[i])

                for j in range(self.num_agents):
                    team_fitness_list = list(map(add, team_fitness_list, agent_fitness_lists[i+j]))

                genome_fitness_lists += [team_fitness_list]

            return genome_fitness_lists

        elif self.reward_level == "individual" and self.team_type == "heterogeneous":
            return agent_fitness_lists

        else:
            raise RuntimeError('Homogeneous-Individual configuration not fully supported yet')

    def generate_model_name(self, fitness):
        pass