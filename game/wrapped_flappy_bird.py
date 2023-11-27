import pygame
import random
import numpy as np
import game.assets_process as assets_process

pygame.init()

'''
加载素材
'''
# 加载图片和音效文件
IMAGES, SOUNDS = assets_process.load_assets()

'''
设置常量
'''
# 游戏屏幕的尺寸，对应素材中背景图片的宽与高
SCREENWIDTH, SCREENHEIGHT = 288, 512
# 游戏帧率
FPS = 30


'''
游戏设置
'''
# 定义屏幕大小
SCREEN = pygame.display.set_mode((SCREENWIDTH, SCREENHEIGHT))
# 定义游戏窗口标题
pygame.display.set_caption('Flappy Bird Demo')
GAMECLOCK = pygame.time.Clock()


class GameState:
    '''
    管理FlappyBird游戏各窗口切换、图形渲染、判断游戏结束条件等功能的类，
    启动游戏也是从这里开始
    '''

    def __init__(self):
        '''
        定义游戏整体的各种状态与参数
        '''
        # 生成一个地板的实例
        self.floor = Floor()
        # 生成一个小鸟的实例
        self.bird = Bird(x=SCREENWIDTH * 0.2, y=SCREENHEIGHT * 0.4)
        # 生成水管列表（管理水管个数、更新的类）
        # 水管列表初始化时，第一批水管就已经被添加进去了，详见PipeManager的__init__方法
        self.pipe_manager = PipeManager()
        # 游戏分数
        self.game_score = 0
        pass

    def game_reset(self):
        '''
        重置游戏的各项参数
        '''
        self.game_score = 0
        self.bird = Bird(x=SCREENWIDTH * 0.2, y=SCREENHEIGHT * 0.4)
        self.pipe_manager = PipeManager()
        pass

    def frame_step(self, action):
        '''
        游戏窗口
        '''

        pygame.event.pump()

        # 初始化这一帧的奖励（reward）和游戏中止的信号变量（terminal）
        reward = 0.1
        terminal = False

        # 更新地板状态（水平左移）
        # 目前地板左移原则上只是提高人的视觉体验，对训练可能起到反效果
        self.floor.update()

        # 根据当前采取的行动(action)，确定小鸟是否拍打翅膀
        flap = True if (action[1] == 1) else False

        # 更新小鸟的状态（切换图片，更改位置等）
        # 根据小鸟在这一时刻是否拍动翅膀，小鸟的位置等信息会有不同的变化
        self.bird.update(flap=flap)

        # 检查这一帧小鸟是不是越过了一对水管。如果是，游戏分数+1，reward变成1
        # 判断小鸟前一帧的左侧、水管中心线与小鸟后一帧左侧的位置关系
        # 这里速度*1.01是为了修bug，不加的话分数无法增加，原因未知，可能和帧数有关
        if self.bird.rect.left + 1.01 * self.pipe_manager.get_first_pipe_up().x_vel < self.pipe_manager.get_first_pipe_up().rect.centerx < self.bird.rect.left:
            SOUNDS['score'].play()
            self.game_score += 1
            reward = 1

        # 更新所有水管的状态（更改位置）
        # 对整个Group使用update()方法，会触发Group内每一个Sprite（水管）的update()方法，水管更改位置的算法写在Pipe类的update()里面，很简单
        self.pipe_manager.pipe_group.update()

        # 如果有水管从左边移出屏幕，把它从列表中删除，在右侧新添一个水管
        # 具体实现过程写在PipeManager的update_pipe_group()方法中
        self.pipe_manager.update_pipe_group()

        # 检查更新位置之后，是否达成了结束游戏的条件（小鸟飞出屏幕、落到地板或碰到水管）。如果是，游戏中止（terminal=True），reward变成-5，重置游戏
        # 目前小鸟原则上不会飞出屏幕
        if self.bird.rect.y > self.floor.y or self.bird.rect.y < 0 or pygame.sprite.spritecollideany(self.bird, self.pipe_manager.pipe_group):
            SOUNDS['hit'].play()
            terminal = True
            reward = -5
            self.game_reset()

        '''
        在画布（屏幕）上绘制各个元素
        绘制顺序：背景、所有水管、地板、分数（可以不绘制）、小鸟
        '''
        SCREEN.blit(IMAGES['bgpic'], (0, 0))

        self.pipe_manager.pipe_group.draw(SCREEN)
        SCREEN.blit(IMAGES['floor'], (self.floor.x, self.floor.y))

        # 在画布上绘制分数
        score_str = str(self.game_score)
        n = len(score_str)
        w = IMAGES['LIST_SCORE']['number_score_00'].get_width() * 1.1
        x = (SCREENWIDTH - n * w) / 2
        y = SCREENHEIGHT * 0.1
        for number in score_str:
            SCREEN.blit(IMAGES['LIST_SCORE']
                        ['number_score_0' + number], (x, y))
            x += w

        SCREEN.blit(self.bird.image, self.bird.rect)

        # 获取这一帧的游戏画面（游戏画面是卷积神经网络的输入成分）
        image_data = pygame.surfarray.array3d(pygame.display.get_surface())

        pygame.display.update()

        # 调整帧速率
        GAMECLOCK.tick(FPS)

        # debug
        print("reward: {}, terminal: {}".format(reward, terminal))

        # 把这一帧的游戏画面、reward、terminal作为参数返回
        return image_data, reward, terminal


class Bird(pygame.sprite.Sprite):
    '''
    小鸟
    '''

    def __init__(self, x, y):
        pygame.sprite.Sprite.__init__(self)
        # 更换小鸟图片使用的参数
        self.img_frames = ['0', '1', '2', '1']
        self.frame_idx = 0
        self.idx = 0
        self.image = IMAGES['LIST_BIRD']['bird0_' + self.img_frames[self.idx]]
        # 图像矩形
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y
        # 小鸟在y轴上的速度，注意正方向为竖直向下
        self.y_vel = -5
        # 重力
        self.gravity = 0.4
        # 小鸟爬升时的各项初始参数
        self.y_vel_after_flap = -6

    def update(self, flap=False):
        '''
        定义小鸟更新自身状态的方法
        '''
        # 如果小鸟爬升，将其在Y轴上的速度更改为初始值
        if flap:
            self.y_vel = self.y_vel_after_flap

        '''
        更新小鸟的速度和y轴坐标
        阻止小鸟从上方飞出屏幕的方法：
        如果小鸟碰到屏幕最上方，将其Y轴速度置为0
        注意Y轴正方向为竖直向下
        '''
        self.y_vel = self.y_vel + self.gravity
        self.rect.y += self.y_vel
        if self.rect.y <= 0:
            self.rect.y = 0
            self.y_vel = self.gravity

        # 每2帧切换一次小鸟的图片
        self.frame_idx = (self.frame_idx+1) % (2*4)
        self.idx = (self.frame_idx % 4)
        self.image = IMAGES['LIST_BIRD']['bird0_' + self.img_frames[self.idx]]


class Pipe(pygame.sprite.Sprite):
    '''
    水管
    '''

    def __init__(self, x, y, upwards=True):
        pygame.sprite.Sprite.__init__(self)
        if upwards:
            self.image = IMAGES['LIST_PIPE']['pipe_up']
            self.rect = self.image.get_rect()
            self.rect.x = x
            self.rect.top = y
        else:
            self.image = IMAGES['LIST_PIPE']['pipe_down']
            self.rect = self.image.get_rect()
            self.rect.x = x
            self.rect.bottom = y
        # 水平速度，向右为正方向
        self.x_vel = -3

    def update(self):
        self.rect.x += self.x_vel


class PipeManager(pygame.sprite.Group):
    '''
    管理水管的类，包含水管列表
    '''
    # 以下的、函数外定义的变量是类的静态变量

    # 水管个数（上下同时出现的一对水管算1个）
    pipe_quantity = 4
    # 水管之间的距离
    pipe_distance = 250
    # 上下水管的间距
    pipe_gap = 125

    def __init__(self, *sprites):
        super().__init__(*sprites)

        # 生成水管实例，一次生成一对水管
        self.pipe_group = pygame.sprite.Group()
        for i in range(self.pipe_quantity):
            pipe_y = random.randint(
                int(SCREENHEIGHT * 0.3), int(SCREENHEIGHT * 0.7))
            new_pipe_up = Pipe(SCREENWIDTH + i *
                               self.pipe_distance, pipe_y, upwards=True)
            new_pipe_down = Pipe(SCREENWIDTH + i * self.pipe_distance, pipe_y -
                                 self.pipe_gap, upwards=False)
            self.pipe_group.add(new_pipe_up)
            self.pipe_group.add(new_pipe_down)

    def get_first_pipe_up(self):
        '''
        获取渲染在画面中的、从左到右第一对水管的上水管
        '''
        return self.pipe_group.sprites()[0]

    def get_first_pipe_down(self):
        '''
        获取渲染在画面中的、从左到右第一对水管的下水管
        '''
        return self.pipe_group.sprites()[1]

    def update_pipe_group(self):
        '''
        如果有水管从左边移出屏幕，把它从列表中删除，在右侧新添一个水管
        '''
        first_pipe_up = self.get_first_pipe_up()
        first_pipe_down = self.get_first_pipe_down()
        if first_pipe_up.rect.right < 0:
            pipe_y = random.randint(
                int(SCREENHEIGHT * 0.3), int(SCREENHEIGHT * 0.7))
            new_pipe_up = Pipe(
                first_pipe_up.rect.x + self.pipe_quantity * self.pipe_distance, pipe_y, upwards=True)
            new_pipe_down = Pipe(first_pipe_up.rect.x + self.pipe_quantity *
                                 self.pipe_distance, pipe_y - self.pipe_gap, upwards=False)
            self.pipe_group.add(new_pipe_up)
            self.pipe_group.add(new_pipe_down)
            first_pipe_up.kill()
            first_pipe_down.kill()
        pass


class Floor(pygame.sprite.Sprite):
    '''
    地板
    '''

    def __init__(self):
        pygame.sprite.Sprite.__init__(self)
        # 地板图片
        self.image = IMAGES['floor']
        # 地板在水平方向上的位置
        self.x = 0
        # 计算地板图片的y值：窗口大小 - 地板图片自身的高度
        self.y = SCREENHEIGHT - self.get_height()

    def update(self):
        '''
        定义地板更新自身状态的方法
        '''
        # 计算地板图片与背景图片的宽度之差
        # 地板x坐标左移，若左移到极限则复位
        floor_gap = self.get_width() - SCREENWIDTH
        self.x -= 4
        if self.x <= -floor_gap:
            self.x = 0

    def get_width(self):
        '''
        获取地板图片的宽度
        '''
        return self.image.get_width()
    
    def get_height(self):
        '''
        获取地板图片的高度
        '''
        return self.image.get_height()


if __name__ == '__main__':
    pass
