import gymnasium as gym
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import load_results
from stable_baselines3.common.results_plotter import ts2xy

from flappybird.env import FlappyBirdEnv


class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    每隔一定时间保存最优模型
    """

    def __init__(self, save_interval=1000):
        super().__init__(verbose=0)

        self.best = -float('inf')
        self.save_interval = save_interval

    def _on_step(self) -> bool:
        if self.n_calls % self.save_interval != 0:
            return True

        # 读取日志
        x, y = ts2xy(load_results('models'), 'timesteps')

        # 求最后100个reward的均值
        mean_reward = sum(y[-100:]) / len(y[-100:])

        # 判断保存
        if mean_reward > self.best:
            self.best = mean_reward
            self.model.save('./runtime_output/models/checkpoint_episode_%d.pth.tar' % self.num_timesteps)

        return True


if __name__ == '__main__':
    # env = FlappyBirdEnv()
    env = gym.make('FlappyBird-v0', render_mode='human')

    # 评估模型
    # model = DQN.load("./runtime_output/models/checkpoint_200000_steps")
    model = DQN.load("./model_20240916_DQN_subproc")
    # model = PPO.load("./model_20240916_PPO_subproc")
    obs, info = env.reset()
    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset()
