import gym
import numpy as np

from gym_TS.agents.DQNAgent import DQNAgent

env = gym.make('gym_TS:TS-v0')
agent = DQNAgent(4,4)
finished_count = []
episodes = 16

for e in range(episodes):
    state = env.reset()
    state = np.reshape(state, [1, 4])
    run_finished = False

    for t in range(1000):
        env.render()
        action = agent.act(state)
        # print("Step "+str(t))
        next_state, reward, done, info = env.step(action)
        next_state = np.reshape(next_state, [1, 4])

        agent.remember(state, action, reward, next_state, done)

        state = next_state

        if done:
            print("episode: {}/{}, time: {}".format(e, episodes, t))
            break

    agent.replay(32)

env.close()
