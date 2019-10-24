"""
Modified from code by Giuseppe Cuccu
"""

import numpy as np
import os
import math
from time import time, sleep
import matplotlib.pyplot as plt
import gym

from gym_TS.agents.TinyAgent import TinyAgent
from sklearn.preprocessing import OneHotEncoder

env = gym.make('gym_TS:TS-v2')
# env = gym.wrappers.Monitor(env, 'video', force = True) # Uncomment to save video

categories = env.get_observation_categories()
one_hot_encoder = OneHotEncoder(sparse=False, categories=categories)

# Training & testing variables
simulation_length = 1000
current_dir = os.getcwd()

# Training variables
training_episodes = 1000
batch_size = simulation_length
display_rate_train = 1
save_rate = 100

# Testing variables
testing_episodes = 100
display_rate_test = 1

# Get size of input and output space and creates agent
observation_size = env.get_observation_size() * len(env.get_observation_categories()[0])  # env.get_observation_size()
action_size = env.get_action_size()

# Get random seed for deterministic fitness (for simplicity)
random_seed = np.random.randint(1e10)


# Fitness function: gameplay loop
def fitness(individual, render=False):
    env.seed(random_seed)  # makes fitness deterministic
    observations = env.reset()
    score = 0
    done = False

    for t in range(simulation_length):
        if render:
            env.render()

        #observation = np.reshape(observation, [1, env.get_observation_size()])
        encoded_observations = [one_hot_encoder.fit_transform(observation.reshape(1, -1)) for observation in observations]

        # All agents act using same controller.
        robot_actions = [individual.act(encoded_observations[i]) for i in range(env.get_num_robots())]

        # The environment changes according to all their actions
        observations, reward, done, info = env.step(robot_actions, t)
        score += reward

        if done:
            break

    return score


# RWG does not distinguish between populations and generations
max_ninds = 100

best_individual = None
max_score = -math.inf

# Neuroevolution loop
for nind in range(max_ninds):
    individual = TinyAgent(observation_size, action_size)
    individual.load_weights()  # No parameters means random weights are generated
    score = fitness(individual)
    print(f"{nind} Score: {score}")
    if score > max_score:
        max_score = score
        best_individual = individual
        fitness(best_individual, render=True)

# Replay winning individual
if best_individual:
    for i in range(10):
        fitness(best_individual, render=True)
