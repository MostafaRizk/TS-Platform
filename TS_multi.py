import gym
import numpy as np
import os
from time import time, sleep
import matplotlib.pyplot as plt

from gym_TS.agents.DQNAgent import DQNAgent
from gym_TS.agents.BasicQAgent import BasicQAgent
from sklearn.preprocessing import OneHotEncoder

env = gym.make('gym_TS:TS-v2')

# Onehotencoding setup
categories_length = max(env.get_num_robots(), env.get_default_num_resources()) + 1
categories = [range(categories_length) for i in range(env.get_state_size())]
one_hot_encoder = OneHotEncoder(sparse=False, categories=categories)
encoded_state_size = env.get_state_size()*categories_length
#print("The size of the state is ", encoded_state_size)
print("The number of possible states is ", len(env.get_possible_states()))

agent = DQNAgent(encoded_state_size, env.get_action_size())
# agent = BasicQAgent(env.get_possible_states(), env.get_action_size())
agent_type = "DQN"  # BasicQ or DQN

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


def train():
    losses = []
    times = []
    rewards = []

    for e in range(training_episodes):
        state = env.reset()

        if agent_type == "DQN":
            state = np.reshape(state, [1, env.get_state_size()])
            state = one_hot_encoder.fit_transform(state)

        total_reward = 0

        for t in range(simulation_length):
            if e % display_rate_train == display_rate_train-1:
                env.render()
                #sleep(0.5)

            # All agents act using same controller.
            robot_actions = [agent.act(state) for i in range(env.get_num_robots())]
            # robot_actions = [env.action_space.sample() for i in range(env.get_num_robots())]

            # The environment changes according to all their actions
            next_state, reward, done, info = env.step(robot_actions, t)

            if agent_type == "DQN":
                next_state = np.reshape(next_state, [1, env.get_state_size()])
                next_state = one_hot_encoder.fit_transform(next_state)

            # The shared controller learns the state,action,reward,next_state tuple for each action.
            for action in robot_actions:
                agent.remember(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward

            if done:
                print("episode: {}/{}, reward: {}, time: {}, ALL RESOURCES COLLECTED".format(e, training_episodes,
                                                                                             total_reward, t))
                times += [t]
                rewards += [total_reward]
                total_reward = 0
                break
            else:
                if t == simulation_length - 1:
                    print("episode: {}/{}, reward: {}".format(e, training_episodes, total_reward))
                    times += [simulation_length + 1]
                    rewards += [total_reward]
                    total_reward = 0

        # Agent learns from experience
        loss = agent.replay(batch_size)
        print("Loss is: " + str(loss))
        losses += [loss]

        # Save model periodically
        if e % save_rate == 0:
            if agent_type == "DQN":
                agent.save(current_dir + '/gym_TS/models/DQN/DQN_Multi_{}_{}.h5'.format(time(), e))
            if agent_type == "BasicQ":
                agent.save(current_dir + '/gym_TS/models/BasicQ/Multi_Q_{}_{}.json'.format(time(), e))

    env.close()

    # Save final model and plot loss/performance
    if agent_type == "DQN":
        agent.save(current_dir+'/gym_TS/models/DQN/DQN_Multi_{}_final.h5'.format(time()))
    if agent_type == "BasicQ":
        agent.save(current_dir + '/gym_TS/models/BasicQ/Multi_Q_{}_final.json'.format(time()))
    plt.subplot(3, 1, 1)
    plt.plot(losses)
    plt.title('Loss, Time taken and Reward')
    plt.ylabel('Loss')

    plt.subplot(3, 1, 2)
    plt.plot(times)
    plt.ylabel('Time Taken')

    plt.subplot(3, 1, 3)
    plt.plot(rewards)
    plt.ylabel('Rewards')

    plt.savefig("v0_loss.png")


def test(filename):
    times = []
    if agent_type == "DQN":
        agent.load_model(current_dir + '/gym_TS/models/DQN/' + filename)  # Load model
    if agent_type == "BasicQ":
        agent.load_model(current_dir + '/gym_TS/models/BasicQ/' + filename)  # Load model

    for e in range(testing_episodes):
        state = env.reset()

        if agent_type == "DQN":
            state = np.reshape(state, [1, env.get_state_size()])
            state = one_hot_encoder.fit_transform(state)

        total_reward = 0

        for t in range(simulation_length):
            if e % display_rate_test == display_rate_test-1:
                env.render()
                #sleep(0.5)

            # All agents act using same controller.
            robot_actions = [agent.act(state) for i in range(env.get_num_robots())]
            # robot_actions = [env.action_space.sample() for i in range(env.get_num_robots())]

            # The environment changes according to all their actions
            next_state, reward, done, info = env.step(robot_actions, t)

            if agent_type == "DQN":
                next_state = np.reshape(next_state, [1, env.get_state_size()])
                next_state = one_hot_encoder.fit_transform(next_state)

            state = next_state
            total_reward += reward

            if done:
                print("episode: {}/{}, reward: {}, time: {}, ALL RESOURCES COLLECTED".format(e, testing_episodes,
                                                                                             total_reward, t))
                times += [t]
                total_reward = 0
                break
            else:
                if t == simulation_length - 1:
                    print("episode: {}/{}, reward: {}".format(e, testing_episodes, total_reward))
                    times += [simulation_length + 1]
                    total_reward = 0

    env.close()

    # Plot results
    plt.ylim([0, simulation_length])
    plt.plot(times)
    plt.savefig("v0_test.png")


#train()
test('DQN_Multi_1571116723.5618603_final.h5')
