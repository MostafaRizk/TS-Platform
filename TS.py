import gym
import time

env = gym.make('gym_TS:TS-v0')
finished_count = []
for i_episode in range(10):
    observation = env.reset()
    for t in range(1000):
        env.render()
        #print(observation)
        action = env.action_space.sample()
        print("Step "+str(t))
        observation, reward, done, info = env.step(action)
        #time.sleep(1)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()