# TODO: 尝试把游戏的一些逻辑（如update各个sprite的部分）进一步封装，使其能够复用于其他游戏
import pygame
import sys
from flappybird.settings import Setting
from flappybird.assets_process import load_sounds
from flappybird.sprites.bird import Bird
from flappybird.sprites.floor import Floor
from flappybird.sprites.pipe import PipeManager
from flappybird.sprites.background import NormalBG, BlackBG
from flappybird.sprites.score import ScoreManager
import flappybird.function


class GameState():
    '''
    管理FlappyBird游戏各窗口切换、图形渲染、判断游戏结束条件等功能的类，
    启动游戏也是从这里开始
    '''

    def __init__(self, setting: Setting):

        # super().__init__()

        # 初始化pygame
        pygame.init()

        # 加载设置
        self.setting = setting

        # 定义游戏屏幕信息
        self.screen = pygame.display.set_mode(
            (self.setting.SCREENWIDTH, self.setting.SCREENHEIGHT))
        pygame.display.set_caption('Flappy Bird Demo')

        # 定义游戏时钟
        self.gameclock = pygame.time.Clock()

        # 加载游戏音效文件
        # TODO: 尝试封装有关游戏音效播放的代码
        self.sounds = load_sounds()

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
            self.sounds['background'].set_volume(0.1)
            self.sounds['background'].play(-1)

    def game_reset(self):
        '''
        重置游戏的各项参数
        '''
        self.score_manager.score = 0
        self.bird = Bird(
            x=self.setting.SCREENWIDTH * 0.2,
            y=self.setting.SCREENHEIGHT * 0.4,
            setting=self.setting)
        self.pipe_manager = PipeManager(setting=self.setting)
        pass

    def frame_step(self, action):
        '''
        执行传入的action并返回这一帧的奖励和训练用图像
        :param action: 传入的动作
        '''

        pygame.event.pump()

        # 初始化这一帧的奖励（reward）和游戏中止的信号变量（terminal）
        reward = 0.1
        terminal = False

        # 根据当前采取的行动(action)，确定小鸟是否拍打翅膀
        if action[1] == 1 and action[0] == 0:
            self.bird.flap = True
            if self.setting.SOUND_PLAY:
                self.sounds['flap'].play()
        elif action[0] == 1 and action[1] == 0:
            self.bird.flap = False
        else:
            print(
                '[dqn_mode_gamestate] Fatal error: gamestate received invalid action!')
            sys.exit(1)

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
        flappybird.function.update(self.floor, self.bird, self.pipe_manager)

        # 检查这一帧小鸟是不是越过了一对水管。如果是，游戏分数+1，reward变成1
        # 判断小鸟前一帧的左侧、水管中心线与小鸟后一帧左侧的位置关系
        # 这里速度*1.01是为了修bug，不加的话分数无法增加，原因未知，可能和帧数有关
        if self.bird.rect.left + 1.01 * self.pipe_manager.get_first_pipe_up(
        ).x_vel < self.pipe_manager.get_first_pipe_up().rect.centerx < self.bird.rect.left:
            self.score_manager.score_increase_1()
            reward = 1
            if self.setting.SOUND_PLAY:
                self.sounds['score'].play()

        # 检查更新位置之后，是否达成了结束游戏的条件（小鸟飞出屏幕、落到地板或碰到水管）。如果是，游戏中止（terminal=True），reward变成-5，重置游戏
        # 目前小鸟原则上不会飞出屏幕
        if self.bird.rect.y > self.floor.y or self.bird.rect.y < 0 or pygame.sprite.spritecollideany(
                self.bird, self.pipe_manager.pipe_group):
            terminal = True
            reward = -5
            # 在控制台打印分数
            if self.setting.PRINT_CONSOLE_LOG:
                print(
                    "[Gamestate] Game over! Score: {}".format(
                        self.score_manager.score))
            self.game_reset()

        """
        在画布（屏幕）上绘制各个元素，供模型训练使用
        绘制顺序：背景、所有水管、地板（静止）、小鸟
        """
        flappybird.function.draw(
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
            flappybird.function.redraw(
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
        return image_data, reward, terminal


if __name__ == '__main__':
    pass
