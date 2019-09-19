import gym
import numpy as np
import os
from time import time

from gym_TS.agents.DQNAgent import DQNAgent

env = gym.make('gym_TS:TS-v1')
agent = DQNAgent(env.get_state_size(), 3)
episodes = 20000
simulation_length = 5000
batch_size = 64
save_rate = 100
display_rate = 100
current_dir = os.getcwd()

for e in range(episodes):
    state = env.reset()
    state = env.one_hot_encode(state)
    state = np.reshape(state, [1, env.get_state_size()])

    # state = env.one_hot_encode(state)
    run_finished = False
    total_reward = 0

    for t in range(simulation_length):
        if e % display_rate == 0: #999:
            env.render()
        # action = env.action_space.sample()
        action = agent.act(state)
        next_state, reward, done, info = env.step(action, t)
        next_state = env.one_hot_encode(next_state)
        next_state = np.reshape(state, [1, env.get_state_size()])
        # next_state = np.reshape(next_state, [1, state_size])
        # next_state = env.one_hot_encode(next_state)

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

    agent.replay(batch_size)

    if e % save_rate == 0:
        agent.save(current_dir + '/gym_TS/models/DQN/DQN_{}_{}.h5'.format(time(), e))
        # agent.generate_q_table(env.get_possible_states())

    test_states = [ np.reshape(env.one_hot_encode(np.array([0, 0, 3])), [1, env.get_state_size()]),
                    np.reshape(env.one_hot_encode(np.array([3, 1, 3])), [1, env.get_state_size()]),
                    np.reshape(env.one_hot_encode(np.array([0, 1, 0])), [1, env.get_state_size()])]
    agent.generate_q_table(test_states)

env.close()
agent.save(current_dir+'/gym_TS/models/DQN/DQN_{}_final.h5'.format(time()))
