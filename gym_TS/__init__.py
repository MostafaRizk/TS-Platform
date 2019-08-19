from gym.envs.registration import register

register(
    id='TS-v0',
    entry_point='gym_TS.envs:TSEnv',
)
