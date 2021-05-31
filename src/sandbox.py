from envs.fork import ForkEnv

env = ForkEnv()
for i in range(100):
    env.render()
    a = input()



