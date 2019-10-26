from gym.envs.registration import register

register(
    id='TS-v2',
    entry_point='gym_TS.envs:TSMultiEnv',
)
