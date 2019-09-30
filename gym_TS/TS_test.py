import gym
import numpy as np
import os
from time import time
import matplotlib.pyplot as plt

from gym_TS.agents.DQNAgent import DQNAgent

env = gym.make('gym_TS:TS-v1')
agent = DQNAgent(env.get_state_size(), 3)
episodes = 1000
simulation_length = 5000
batch_size = 64
save_rate = 100
display_rate = 10000000
current_dir = os.getcwd()

def train():
    losses = []

    for e in range(episodes):
        state = env.reset()
        state = env.one_hot_encode(state)
        state = np.reshape(state, [1, env.get_state_size()])

        total_reward = 0

        for t in range(simulation_length):
            if e % display_rate == display_rate-1:
                env.render()
            action = agent.act(state)
            next_state, reward, done, info = env.step(action, t)
            next_state = env.one_hot_encode(next_state)
            next_state = np.reshape(next_state, [1, env.get_state_size()])

            total_reward += reward

            agent.remember(state, action, reward, next_state, done)

            state = next_state

            if done:
                print("episode: {}/{}, reward: {}, time: {}".format(e, episodes, total_reward, t))
                total_reward = 0
                break
            else:
                if t == simulation_length-1:
                    print("episode: {}/{}, reward: {}, FAILED".format(e, episodes, total_reward))
                    total_reward = 0

        loss = agent.replay(batch_size)
        print("Loss is: " + str(loss))
        losses += [loss]

        if e % save_rate == 0:
            agent.save(current_dir + '/gym_TS/models/DQN/DQN_{}_{}.h5'.format(time(), e))

    env.close()
    agent.save(current_dir+'/gym_TS/models/DQN/DQN_{}_final.h5'.format(time()))
    plt.plot(losses)
    plt.savefig("v1_loss.png")

def test():
    for e in range(episodes):
        state = env.reset()
        state = env.one_hot_encode(state)
        state = np.reshape(state, [1, env.get_state_size()])

        total_reward = 0

        for t in range(simulation_length):
            if e % display_rate == display_rate - 1:
                env.render()
            action = agent.act(state)
            next_state, reward, done, info = env.step(action, t)
            next_state = env.one_hot_encode(next_state)
            next_state = np.reshape(next_state, [1, env.get_state_size()])

            state = next_state

            if done:
                print("episode: {}/{}, reward: {}, time: {}".format(e, episodes, total_reward, t))
                total_reward = 0
                break
            else:
                if t == simulation_length - 1:
                    print("episode: {}/{}, reward: {}, FAILED".format(e, episodes, total_reward))
                    total_reward = 0

        if e % save_rate == 0:
            agent.save(current_dir + '/gym_TS/models/DQN/DQN_{}_{}.h5'.format(time(), e))

    env.close()

train()

