import gym
import numpy as np
import os
from time import time
import matplotlib.pyplot as plt

from gym_TS.agents.DQNAgent import DQNAgent
from gym_TS.agents.BasicQAgent import BasicQAgent

env = gym.make('gym_TS:TS-v0')  # Simple
agent = DQNAgent(env.get_state_size(), 2)  # Simple

#BasicQ
#agent = BasicQAgent(env.get_possible_states(), 2)

#env = gym.make('gym_TS:TS-v1') # Normal
#agent = DQNAgent(env.get_state_size(), 3)  # Normal
training_episodes = 200
testing_episodes = 100
simulation_length = 5000
batch_size = simulation_length # 32
save_rate = 100
display_rate_train = 1000
display_rate_test = 1
current_dir = os.getcwd()


def train_basic_agent():
    losses = []
    times = []

    for e in range(training_episodes):
        state = env.reset()

        total_reward = 0

        for t in range(simulation_length):
            if e % display_rate_train == display_rate_train-1:
                env.render()
            elif e > training_episodes * 0.8:
                env.render()
            action = agent.act(state)
            next_state, reward, done, info = env.step(action, t)

            total_reward += reward
            agent.remember(state, action, reward, next_state, done)
            state = next_state

            if done:
                print("episode: {}/{}, reward: {}, time: {}".format(e, training_episodes, total_reward, t))
                times += [t]
                total_reward = 0
                break
            else:
                if t == simulation_length-1:
                    print("episode: {}/{}, reward: {}, FAILED".format(e, training_episodes, total_reward))
                    times += [simulation_length + 1]
                    total_reward = 0

        loss = agent.replay(batch_size)
        print("Loss is: " + str(loss))
        losses += [loss]

        if e % save_rate == 0:
            agent.save(current_dir + '/gym_TS/models/BasicQ/Q_{}_{}.json'.format(time(), e))

    env.close()
    agent.save(current_dir+'/gym_TS/models/BasicQ/Q_{}_final.json'.format(time()))
    plt.subplot(2, 1, 1)
    plt.plot(losses)
    plt.title('Loss and Time taken')
    plt.ylabel('Loss')

    plt.subplot(2, 1, 2)
    plt.plot(times)
    plt.ylabel('Time Taken')
    plt.savefig("v0_loss.png")


def train():
    losses = []
    times = []

    for e in range(training_episodes):
        state = env.reset()
        state = env.one_hot_encode(state)
        state = np.reshape(state, [1, env.get_state_size()])

        total_reward = 0

        for t in range(simulation_length):
            if e % display_rate_train == display_rate_train-1:
                env.render()
            action = agent.act(state)
            next_state, reward, done, info = env.step(action, t)
            next_state = env.one_hot_encode(next_state)
            next_state = np.reshape(next_state, [1, env.get_state_size()])

            total_reward += reward

            agent.remember(state, action, reward, next_state, done)

            state = next_state

            if done:
                print("episode: {}/{}, reward: {}, time: {}".format(e, training_episodes, total_reward, t))
                times += [t]
                total_reward = 0
                break
            else:
                if t == simulation_length-1:
                    print("episode: {}/{}, reward: {}, FAILED".format(e, training_episodes, total_reward))
                    times += [simulation_length + 1]
                    total_reward = 0

        loss = agent.replay(batch_size)
        print("Loss is: " + str(loss))
        losses += [loss]

        if e % save_rate == 0:
            agent.save(current_dir + '/gym_TS/models/DQN/DQN_{}_{}.h5'.format(time(), e))

    env.close()
    agent.save(current_dir+'/gym_TS/models/DQN/DQN_{}_final.h5'.format(time()))
    plt.subplot(2, 1, 1)
    plt.plot(losses)
    plt.title('Loss and Time taken')
    plt.ylabel('Loss')

    plt.subplot(2, 1, 2)
    plt.plot(times)
    plt.ylabel('Time Taken')
    plt.savefig("v0_loss.png")


def test(filename):
    times = []
    agent.load_model(current_dir + '/gym_TS/models/DQN/' + filename)  # Load model

    for e in range(testing_episodes):
        state = env.reset()
        state = env.one_hot_encode(state)
        state = np.reshape(state, [1, env.get_state_size()])

        total_reward = 0

        for t in range(simulation_length):
            if e % display_rate_test == display_rate_test - 1:
                env.render()

            action = agent.act(state)  # act on state based on model
            next_state, reward, done, info = env.step(action, t)
            next_state = env.one_hot_encode(next_state)
            next_state = np.reshape(next_state, [1, env.get_state_size()])
            state = next_state

            if done:
                print("episode: {}/{}, reward: {}, time: {}".format(e, testing_episodes, total_reward, t))
                times += [t]
                total_reward = 0
                break
            else:
                if t == simulation_length - 1:
                    print("episode: {}/{}, reward: {}, FAILED".format(e, testing_episodes, total_reward))
                    times += [simulation_length + 1]
                    total_reward = 0

    env.close()
    plt.plot(times)
    plt.savefig("v0_test.png")


def test_basic_agent(filename):
    times = []
    agent.load(current_dir + '/gym_TS/models/BasicQ/' + filename)  # Load model

    for e in range(testing_episodes):
        state = env.reset()

        total_reward = 0

        for t in range(simulation_length):
            if e % display_rate_test == display_rate_test - 1:
                env.render()

            action = agent.act(state)  # act on state based on model
            next_state, reward, done, info = env.step(action, t)
            state = next_state

            if done:
                print("episode: {}/{}, reward: {}, time: {}".format(e, testing_episodes, total_reward, t))
                times += [t]
                total_reward = 0
                break
            else:
                if t == simulation_length - 1:
                    print("episode: {}/{}, reward: {}, FAILED".format(e, testing_episodes, total_reward))
                    times += [simulation_length + 1]
                    total_reward = 0

    env.close()
    plt.ylim([0, simulation_length])
    plt.plot(times)
    plt.savefig("v0_test.png")

#train()
test('v0_2.h5')
#test('DQN_1570063893.753158_0.h5')

#train_basic_agent()
#agent.display()
#test_basic_agent('Q_1570086600.6342318_final.json')
#agent.display()



