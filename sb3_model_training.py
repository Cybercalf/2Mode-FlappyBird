import gymnasium as gym

from multiprocessing import freeze_support

from stable_baselines3 import DQN, PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
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


def make_train_env():
    def _call():
        env = gym.make('FlappyBird-v0', render_mode='none')
        return env
    return _call


if __name__ == '__main__':

    freeze_support()

    # env = FlappyBirdEnv()
    # env = gym.make('FlappyBird-v0', render_mode='none')

    env = SubprocVecEnv([make_train_env() for _ in range(10)])

    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path='./runtime_output/models/',
        name_prefix='checkpoint',
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    # # 从头训练模型
    model = DQN('CnnPolicy', env, buffer_size=5000, verbose=0, tensorboard_log='./runtime_output/tensorboard/dqn_flappybird/')
    model.learn(
        total_timesteps=10000,
        progress_bar=True,
        tb_log_name='20240916_DQN_subproc',
        reset_num_timesteps=False,
        callback=checkpoint_callback
    )
    model.save('model_20240916_DQN_subproc')

    # model = PPO('MlpPolicy', env, verbose=0, tensorboard_log='./runtime_output/tensorboard/ppo_flappybird/')
    # model.learn(
    #     total_timesteps=1000000,
    #     progress_bar=True,
    #     tb_log_name='20240916_PPO_subproc',
    #     reset_num_timesteps=False,
    # )
    # model.save('model_20240916_PPO_subproc')

    # 从已有模型开始训练
    # model = DQN.load('./runtime_output/models/checkpoint_138000_steps', env, verbose=0, tensorboard_log='./runtime_output/dqn_flappybird_tensorboard/')
    # model.learn(total_timesteps=62000,
    #             progress_bar=True,
    #             tb_log_name='DQN_resume',
    #             reset_num_timesteps=False,
    #             callback=checkpoint_callback)
    # model.save('test_model_resume')

    # # 评估模型
    # model = DQN.load("./runtime_output/models/checkpoint_200000_steps")
    # obs, info = env.reset()
    # while True:
    #     action, _states = model.predict(obs, deterministic=True)
    #     obs, reward, terminated, truncated, info = env.step(action)
    #     if terminated or truncated:
    #         obs, info = env.reset()
