import gym
import os
import gym_TS

from ..algorithm.parameters import params
from ..fitness.base_ff_classes.base_ff import base_ff
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
        calculator = fitness_calculator.FitnessCalculator()  # Get random seed and simulation length from parameters
        individual = GEAgent(ind)
        fitness_score = calculator.calculate_fitness()
        return fitness_score
