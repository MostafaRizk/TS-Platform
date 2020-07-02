from fitness import FitnessCalculator
from learning.rwg import RWGLearner

parameter_filename = 'default_parameters.json'
fitness_calculator = FitnessCalculator(parameter_filename)
learner = RWGLearner(fitness_calculator)

genome, fitness = learner.learn()

print(f"Fitness is {fitness}")

