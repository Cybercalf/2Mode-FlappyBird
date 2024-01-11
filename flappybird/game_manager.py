import pygame
import sys

from .settings import Setting
from .sprite.bird import Bird
from .sprite.floor import Floor
from .sprite.pipe import PipeManager
from .sprite.background import NormalBG, BlackBG
from .sprite.score import ScoreManager
from .sound_manager import SoundManager
from .window.end_window import EndWindow
from .util import function as function
from .util.custom_exception import InvalidActionException, InvalidPlayerException


class GameManager():
    '''
    管理FlappyBird游戏各窗口切换、图形渲染、判断游戏结束条件等功能的类，
    启动游戏也是从这里开始
    '''

    def __init__(self, setting: Setting):

        # super().__init__()

        # 初始化pygame
        pygame.init()

        # 指定游戏玩家
        self.player = 'human'

        # 加载设置
        self.setting = setting

        # 定义游戏屏幕信息
        self.screen = pygame.display.set_mode(
            (self.setting.SCREENWIDTH, self.setting.SCREENHEIGHT))
        pygame.display.set_caption('Flappy Bird Demo')

        # 定义游戏时钟
        self.gameclock = pygame.time.Clock()

        # 加载游戏音效文件
        self.sounds = SoundManager()

        # 游戏窗口
        self.end_window = EndWindow(self.setting)

        # 游戏是否中止
        self.terminated = False

        """
        生成实例：游戏背景、地板、小鸟
        生成实例：水管管理类和游戏分数管理类
        水管列表初始化时，第一批水管就已经被添加进去了，详见PipeManager的__init__()方法
        """
        self.normal_bg = NormalBG()
        self.black_bg = BlackBG()
        self.floor = Floor(setting=setting)
        self.bird = Bird(
            x=self.setting.SCREENWIDTH * 0.2,
            y=self.setting.SCREENHEIGHT * 0.4,
            setting=setting)
        self.pipe_manager = PipeManager(setting=setting)
        self.score_manager = ScoreManager(setting=setting)

        if self.setting.SOUND_PLAY:
            self.sounds.play('background', volume=0.1, loop=True)

    def game_reset(self):
        '''
        重置游戏的各项参数
        '''
        self.score_manager.reset_score()
        self.bird = Bird(
            x=self.setting.SCREENWIDTH * 0.2,
            y=self.setting.SCREENHEIGHT * 0.4,
            setting=self.setting)
        self.pipe_manager = PipeManager(setting=self.setting)
        self.terminated = False

    def set_player_human(self):
        '''
        指定游戏玩家为人类
        '''
        self.player = 'human'

    def set_player_computer(self):
        '''
        指定游戏玩家为电脑
        '''
        self.player = 'computer'

    def load_setting(self, setting):
        '''
        加载游戏设置
        :param setting: 要加载的设置
        :return 加载的设置，与传入的setting相同
        '''
        self.setting = setting
        return setting

    def get_current_score(self):
        '''
        获取当前游戏的分数
        '''
        return self.score_manager.get_score()

    def is_terminated(self):
        '''
        检查当前游戏是否中止
        '''
        return self.terminated

    def start_game_by_human(self):
        '''
        在游戏玩家为人类的情况下启动游戏
        '''
        self.set_player_human()
        while True:
            while True:
                _, _, _ = self.frame_step()
                if self.is_terminated():
                    if self.setting.SOUND_PLAY:
                        self.sounds.play('hit')
                    break
            self.end_window.show(self.screen)

    def frame_step(self, action=[0, 0]):
        '''
        执行传入的action并返回这一帧的奖励和训练用图像
        :param action: 传入的动作
        '''

        # 如果游戏中止，重置游戏数据再进行游戏
        if self.is_terminated():
            self.game_reset()

        pygame.event.pump()

        # 初始化这一帧的奖励（reward）
        reward = 0.1

        """
        确定小鸟此刻的动作
        若游戏玩家为电脑，则根据传入的action确定动作
        若游戏玩家为人类，则监测键盘输入
        """
        if self.player == 'computer':
            # 根据当前采取的行动(action)，确定小鸟是否拍打翅膀
            if action[1] == 1 and action[0] == 0:
                self.bird.flap = True
            elif action[0] == 1 and action[1] == 0:
                self.bird.flap = False
            else:
                # 如果检测到非法的action传入，生成异常
                # 目前该错误理论上不会被触发
                raise InvalidActionException('When model playing game, an invalid action is received.')
                sys.exit(1)
        elif self.player == 'human':
            # 通过pygame.event模块，不断获取当前发生的事件
            # pygame启动后输入法可能被偷偷调整为中文，导致字母按下没有反应。
            # 启动游戏后若出现此问题，可以用快捷键切输入法中英文后重试
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    quit()
                if event.type == pygame.KEYDOWN:
                    # 按下空格使小鸟爬升
                    if event.key == pygame.K_SPACE:
                        self.bird.flap = True
        else:
            # 若游戏玩家非法，生成异常
            raise InvalidPlayerException('Invalid game player.')
            sys.exit(1)

        # 若小鸟此刻拍动翅膀，且设置规定游戏播放声音，则播放拍翅膀的音效
        if self.bird.flap and self.setting.SOUND_PLAY:
            self.sounds.play('flap')

        """
        更新所有元素的位置
        1.更新地板状态（水平左移）
        目前地板左移原则上只是提高人的视觉体验，对训练可能起到反效果
        目前解决方案：传入网络的地板图像是静止的，只有显示给人看的地板是运动的
        2.更新小鸟的状态（切换图片，更改位置等）
        根据小鸟在这一时刻是否拍动翅膀，小鸟的位置等信息会有不同的变化
        3.更新所有水管的状态（更改位置）
        如果有水管从左边移出屏幕，把它从列表中删除，在右侧新添一个水管
        具体实现过程写在PipeManager的update_pipe_group()方法中
        """
        function.update(self.floor, self.bird, self.pipe_manager)

        # 检查这一帧小鸟是不是越过了一对水管。如果是，游戏分数+1，reward变成1
        # 判断小鸟前一帧的左侧、水管中心线与小鸟后一帧左侧的位置关系
        # 这里速度*1.01是为了修bug，不加的话分数无法增加，原因未知，可能和帧数有关
        if self.bird.rect.left + 1.01 * self.pipe_manager.get_first_pipe_up(
        ).x_vel < self.pipe_manager.get_first_pipe_up().rect.centerx < self.bird.rect.left:
            self.score_manager.update_score()
            reward = 1
            if self.setting.SOUND_PLAY:
                self.sounds.play('score')

        # 检查更新位置之后，是否达成了结束游戏的条件（小鸟飞出屏幕、落到地板或碰到水管）。如果是，游戏中止，reward变成-5，重置游戏
        # 目前小鸟原则上不会飞出屏幕
        if self.bird.rect.y > self.floor.rect.y or self.bird.rect.y < 0 or pygame.sprite.spritecollideany(
                self.bird, self.pipe_manager.pipe_group):
            reward = -5
            self.terminated = True

        """
        在画布（屏幕）上绘制各个元素，供模型训练使用
        绘制顺序：背景、所有水管、地板（静止）、小鸟
        """
        function.draw(
            self.black_bg,
            self.pipe_manager,
            self.floor.get_still_floor(),
            self.bird,
            surface=self.screen)

        # 获取这一帧的游戏画面（游戏画面是卷积神经网络的输入成分）
        image_data = pygame.surfarray.array3d(pygame.display.get_surface())

        if not self.setting.SHOW_RL_OBSERVATION_SCREEN:
            """
            在演示模式下重置屏幕，在画布（屏幕）上绘制各个元素，供人观看
            绘制顺序：背景、所有水管、地板、分数、小鸟
            """
            function.redraw(
                self.normal_bg,
                self.pipe_manager,
                self.floor,
                self.score_manager,
                self.bird,
                surface=self.screen)

        pygame.display.update()

        # 调整帧速率
        self.gameclock.tick(self.setting.FPS)

        # 把这一帧的游戏画面、reward、terminal作为参数返回
        return image_data, reward, self.terminated


if __name__ == '__main__':
    pass
