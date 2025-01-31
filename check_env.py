from stable_baselines3.common.env_checker import check_env

from flappybird.env import FlappyBirdEnv

if __name__ == '__main__':
    env = FlappyBirdEnv()
    check_env(env)
