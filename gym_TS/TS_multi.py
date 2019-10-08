import gym
import numpy as np
import os
from time import time
import matplotlib.pyplot as plt

from gym_TS.agents.DQNAgent import DQNAgent
from gym_TS.agents.BasicQAgent import BasicQAgent

env = gym.make('gym_TS:TS-v2')  # Simple
#agent = DQNAgent(env.get_state_size(), 2)

training_episodes = 10
testing_episodes = 100
simulation_length = 5000
batch_size = simulation_length
save_rate = 100
display_rate_train = 1000
display_rate_test = 1
current_dir = os.getcwd()


def train():
    losses = []
    times = []

    for e in range(training_episodes):
        state = env.reset()
        print(state)

        for t in range(simulation_length):
            if e % display_rate_train == display_rate_train-1:
                env.render()

            # Agent acts, observes the new state and remembers the outcome
            '''
            action = agent.act(state)
            next_state, reward, done, info = env.step(action, t)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            '''
            done = False

            if done:
                print("episode: {}/{}, time: {}".format(e, training_episodes, t))
                times += [t]
                break
            else:
                if t == simulation_length-1:
                    print("episode: {}/{}, FAILED".format(e, training_episodes))
                    times += [simulation_length + 1]

        # Agent learns from experience
        '''
        loss = agent.replay(batch_size)
        print("Loss is: " + str(loss))
        losses += [loss]
        '''

        # Save model periodically
        '''
        if e % save_rate == 0:
            agent.save(current_dir + '/gym_TS/models/DQN/DQN_{}_{}.h5'.format(time(), e))
        '''

    env.close()

    # Save final model and plot loss/performance
    '''
    agent.save(current_dir+'/gym_TS/models/DQN/DQN_{}_final.h5'.format(time()))
    plt.subplot(2, 1, 1)
    plt.plot(losses)
    plt.title('Loss and Time taken')
    plt.ylabel('Loss')

    plt.subplot(2, 1, 2)
    plt.plot(times)
    plt.ylabel('Time Taken')
    plt.savefig("v0_loss.png")
    '''

train()





