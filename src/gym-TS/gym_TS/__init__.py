from gym.envs.registration import register

register(
    id='TS-v1',
    entry_point='src.gym_TS.envs:SlopeEnvGym',
)
