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

env = gym.make('gym_TS:TS-v2')
# env = gym.wrappers.Monitor(env, 'video', force = True) # Uncomment to save video

# Training & testing variables
simulation_length = 5000
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
observation_size = env.get_state_size()
action_size = env.get_action_size()

# Get random seed for deterministic fitness (for simplicity)
random_seed = np.random.randint(1e10)


# Fitness function: gameplay loop
def fitness(individual, render=False):
    env.seed(random_seed)  # makes fitness deterministic
    observation = env.reset()
    score = 0
    done = False

    for t in range(simulation_length):
        if render:
            env.render()

        observation = np.reshape(observation, [1, env.get_state_size()])

        # All agents act using same controller.
        robot_actions = [individual.act(observation) for i in range(env.get_num_robots())]

        # The environment changes according to all their actions
        observation, reward, done, info = env.step(robot_actions, t)
        score += reward

        if done:
            break

    print(f"Score: {score}")

    return score


# RWG does not distinguish between populations and generations
max_ninds = 1000

best_individual = None
max_score = -math.inf

# Neuroevolution loop
for nind in range(max_ninds):
    individual = TinyAgent(observation_size, action_size)
    individual.load_weights()  # No parameters means random weights are generated
    score = fitness(individual)
    if score > max_score:
        max_score = score
        best_individual = individual

# Replay winning individual
if best_individual:
    fitness(best_individual, render=True)
