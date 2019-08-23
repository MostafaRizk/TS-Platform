import gym
import time

env = gym.make('gym_TS:TS-v1')
finished_count = []
for i_episode in range(10):
    observation = env.reset()
    run_finished = False
    for t in range(1000):
        env.render()
        #print(observation)
        action = env.action_space.sample()
        print("Step "+str(t))
        observation, reward, done, info = env.step(action)
        #time.sleep(1)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            finished_count += [t]
            run_finished = True
            break
    if not run_finished:
        finished_count += ["N/A"]
env.close()
for run in finished_count:
    print(run)