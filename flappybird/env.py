import PIL.Image
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box, Discrete

from flappybird.settings import Setting
from flappybird.game_manager import GameManager


class FlappyBirdEnv(gym.Env):
    """
    符合gym规范的游戏环境
    """

    # 初始状态
    """
    初始化一个数组，尺寸128*72，数据类型np.uint8
    初始化一个“状态”，由四个数组在axis=0处叠加得到
    empty_frame代表一帧画面，empty_state代表由连续的4帧组成的一个状态
    """
    empty_frame = np.zeros((128, 72), dtype=np.uint8)
    empty_state = np.stack(
        (empty_frame, empty_frame, empty_frame, empty_frame),
        axis=0
    )

    # 环境元数据
    metadata = {'render_modes': ['human', 'raw', None]}

    def __init__(self, render_mode=None):
        """
        初始化方法
        """

        """
        指定环境的渲染方式
        """
        self.render_mode = render_mode

        # 初始化游戏
        game_setting = Setting()
        if self.render_mode == 'human':
            game_setting.set_render_mode('human')
        elif self.render_mode == 'raw':
            game_setting.set_render_mode('raw')
        else:
            game_setting.set_render_mode('hidden')
        self.game = GameManager(game_setting)
        self.game.set_player_computer()

        # 初始化当前状态
        self.current_state = FlappyBirdEnv.empty_state

        # The Space object corresponding to valid actions, all valid actions should be contained within the space.
        """
        两个离散动作：
        0: 不拍翅膀
        1: 拍翅膀
        """
        self.action_space = Discrete(2)

        # The Space object corresponding to valid observations, all valid observations should be contained within the space.
        """
        Observation: 4张128*72的灰度图像
        """
        self.observation_space = Box(low=0,
                                     high=255,
                                     shape=(4, 128, 72),
                                     dtype=np.uint8)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        '''
        Resets the environment to an initial state, required before calling step. Returns the first agent observation for an episode and information, i.e. metrics, debug info.
        '''
        self.game.game_reset()
        self.current_state = FlappyBirdEnv.empty_state

        info = {}

        # 返回observation, info
        return self.current_state, info

    def step(self, action):
        '''
        Updates an environment with actions returning the next agent observation, the reward for taking that actions, if the environment has terminated or truncated due to the latest action and information from the environment about the step, i.e. metrics, debug info.
        '''
        action_to_game = [0, 0]
        action_to_game[action] = 1

        # 把动作传入游戏，得到新观测到的一帧图像、奖励值、游戏是否结束
        o_frame, reward, terminated = self.game.frame_step(action_to_game)

        '''
        对输入的帧图像做预处理

        输入图像：512*288，rgb彩色图像

        预处理步骤：

        1.降采样，使图像尺寸变为128*72

        2.将图像转换为灰度图像
        '''
        downsample_frame = PIL.Image.fromarray(o_frame).resize((72, 128)).convert(mode='L')
        output_frame = np.asarray(downsample_frame).astype(np.uint8)
        # output_frame[output_frame <= 1.] = 0.0
        # output_frame[output_frame > 1.] = 1.0
        # output_frame = output_frame.astype(np.uint8)

        """
        更新当前状态，用于返回observation
        把当前state（4帧图像）最早的一帧去除，加上从游戏得到的最新的一帧图像，作为下一个state
        """
        self.current_state = np.append(
            self.current_state[1:, :, :], output_frame.reshape((1,) + output_frame.shape),
            axis=0
        )

        # 返回observation, reward, terminated, truncated, info
        return self.current_state, reward, terminated, False, {}

    def render(self):
        '''
        Renders the environments to help visualise what the agent see, examples modes are “human”, “rgb_array”, “ansi” for text.
        '''
        pass
