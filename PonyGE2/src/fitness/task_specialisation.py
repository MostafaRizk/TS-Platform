import gym
import os
import gym_TS

from algorithm.parameters import params
from fitness.base_ff_classes.base_ff import base_ff
from gym_TS import fitness_calculator
from gym_TS.agents import GEAgent

class task_specialisation(base_ff):
    """Calculates fitness according to the function defined in the ARGoS Loop Functions"""

    maximise = True
    dots = None

    def __init__(self):
        # Initialise base fitness function class.
        super().__init__()
        self.env = gym.make('gym_TS:TS-v2')

    def escape_characters(self, escape_string):
        escape_string = escape_string.replace(";","\;")
        return escape_string

    def evaluate(self, ind, **kwargs):
        individual = GEAgent(ind)
        simulation_length = 1000 #TODO Update to use params
        num_trials = 3 #TODO Update to use params
        calculator = fitness_calculator.FitnessCalculator(output_selection_method="argmax", random_seed=params['RANDOM_SEED'], simulation_length=simulation_length)  # Get random seed and simulation length from parameters
        fitness_score = calculator.calculate_fitness(individual, num_trials=num_trials, render=False, learning_method="GE")
        return fitness_score
