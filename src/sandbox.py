import argparse
import json

from agents.hardcoded.collector import HardcodedCollectorAgent
from agents.hardcoded.dropper import HardcodedDropperAgent
from agents.hardcoded.generalist import HardcodedGeneralistAgent
from agents.hardcoded.lazy_generalist import HardcodedLazyGeneralistAgent
from fitness import FitnessCalculator
from learning.cma import CMALearner
from learning.rwg import RWGLearner
import numpy as np
from agents.nn_agent_lean import NNAgent
import pandas as pd

parameter_filename = "default_parameters.json"
fitness_calculator = FitnessCalculator(parameter_filename)

learner = CMALearner(fitness_calculator)
genome, fitness = learner.learn()