from gymnasium.envs.registration import register

register(
    id='FlappyBird-v0',
    entry_point='flappybird.env:FlappyBirdEnv'
)
