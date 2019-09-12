import gym
import numpy as np
import os
from time import time

from gym_TS.agents.DQNAgent import DQNAgent

env = gym.make('gym_TS:TS-v0')
agent = DQNAgent(env.get_num_states(), 3)
finished_count = []
episodes = 20000
current_dir = os.getcwd()

for e in range(episodes):
    state = env.reset()
    state = np.reshape(state, [1, 4])
    state = env.one_hot_encode(state)
    run_finished = False
    total_reward = 0

    for t in range(1000):
        if e % 100 == 0:
            env.render()
        # action = env.action_space.sample()
        action = agent.act(state)
        next_state, reward, done, info = env.step(action, t)
        next_state = np.reshape(next_state, [1, 4])
        next_state = env.one_hot_encode(next_state)

        total_reward += reward

        agent.remember(state, action, reward, next_state, done)

        state = next_state

        if done:
            print("episode: {}/{}, reward: {}, time: {}".format(e, episodes, total_reward, t))
            total_reward = 0
            break
        else:
            if t==999:
                print("episode: {}/{}, reward: {}, FAILED".format(e, episodes, total_reward))
                total_reward = 0

    agent.replay(64)

    if e % 100 == 0:
        agent.save(current_dir + '/gym_TS/models/DQN/DQN_{}_{}.h5'.format(time(), e))

env.close()
agent.save(current_dir+'/gym_TS/models/DQN/DQN_{}_final.h5'.format(time()))
