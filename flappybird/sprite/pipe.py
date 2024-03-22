import pygame
import random
from ..util.interface import GameElement, GameSprite


class Pipe(GameSprite):
    '''
    水管
    '''

    def __init__(self, x, y, render_setting, upwards=True):
        """
        :param x: 水管横坐标
        :param y: 水管纵坐标
        :param upwards: 水管口朝向。True: 上; False: 下
        """
        super().__init__()
        if upwards:
            self.image = pygame.image.load('flappybird/assets/sprites/pipe_up.png')
            self.rect = self.image.get_rect()
            self.rect.x = x
            self.rect.top = y
        else:
            self.image = pygame.image.load('flappybird/assets/sprites/pipe_down.png')
            self.rect = self.image.get_rect()
            self.rect.x = x
            self.rect.bottom = y
        # 水平速度，向右为正方向
        # 注意速度时间单位为1帧的时间
        self.x_vel = -3 * 60 / render_setting.FPS

    def update(self):
        self.rect.x += self.x_vel


class PipeManager(GameElement):
    '''
    管理水管的类，包含水管列表
    '''
    # 以下的、函数外定义的变量是类的静态变量

    # 初始生成的水管个数（上下同时出现的一对水管算1个）
    init_pipe_quantity = 4

    def __init__(self, render_setting, sprite_setting):
        super().__init__()

        self.render_setting = render_setting
        if sprite_setting is not None:
            self.sprite_setting = sprite_setting
        else:
            self.sprite_setting = {'distance': 150, 'gap': 130}

        # 水管之间的距离
        # 当水管间距离为0时，相邻水管会紧贴在一起，但不会重合
        # self.pipe_distance = 150
        self.pipe_distance = self.sprite_setting.get('distance', 150)
        # 上下水管的间距
        # self.pipe_gap = 130
        self.pipe_gap = self.sprite_setting.get('gap', 130)

        # 生成存储水管的容器
        self.pipe_group = pygame.sprite.Group()

        # 生成水管并放入容器
        example_pipe_width = pygame.image.load('flappybird/assets/sprites/pipe_up.png').get_width()
        for i in range(self.init_pipe_quantity):
            pipe_y = random.randint(
                int(self.render_setting.SCREENHEIGHT * 0.3), int(self.render_setting.SCREENHEIGHT * 0.7))
            new_pipe_up = Pipe(self.render_setting.SCREENWIDTH + i * (self.pipe_distance + example_pipe_width),
                               pipe_y, render_setting=self.render_setting, upwards=True)
            new_pipe_down = Pipe(self.render_setting.SCREENWIDTH + i * (self.pipe_distance + example_pipe_width),
                                 pipe_y - self.pipe_gap, render_setting=self.render_setting, upwards=False)
            self.pipe_group.add(new_pipe_up)
            self.pipe_group.add(new_pipe_down)

    def get_pipe_pair_quantity(self):
        '''
        获取已经生成的水管数量(单位：对)(上下同时出现的一对水管算1个)
        '''
        return len(self.pipe_group.sprites()) // 2

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

    def get_last_pipe_up(self):
        '''
        获取已经生成的水管中，从左到右最后一对水管的上水管
        '''
        return self.pipe_group.sprites()[-2]

    def get_last_pipe_down(self):
        '''
        获取已经生成的水管中，从左到右最后一对水管的下水管
        '''
        return self.pipe_group.sprites()[-1]

    def update_pipe_group(self):
        '''
        1.移动每一根水管的位置（调用每一个Pipe的update()方法）
        2.如果有水管从左边移出屏幕，把它从列表中删除
        3.如果已生成的最后一对水管已经出现在屏幕中，再在其右侧生成一对水管
        '''
        # 对整个Group使用update()方法，会触发Group内每一个Sprite（水管）的update()方法，水管更改位置的算法写在Pipe类的update()里面，很简单
        self.pipe_group.update()
        # 如果有水管从左边移出屏幕，把它从列表中删除，在右侧新添一个水管
        first_pipe_up = self.get_first_pipe_up()
        first_pipe_down = self.get_first_pipe_down()
        if first_pipe_up.rect.right < 0:
            first_pipe_up.kill()
            first_pipe_down.kill()
        # 如果已生成的最后一对水管已经出现在屏幕中，再在其右侧生成一对水管
        last_pipe_up = self.get_last_pipe_up()
        if last_pipe_up.rect.left < self.render_setting.SCREENWIDTH:
            current_pipe_pair_quantity = self.get_pipe_pair_quantity()
            pipe_y = random.randint(
                int(self.render_setting.SCREENHEIGHT * 0.3), int(self.render_setting.SCREENHEIGHT * 0.7))
            new_pipe_up = Pipe(first_pipe_up.rect.x + current_pipe_pair_quantity * (self.pipe_distance + first_pipe_up.rect.width),
                               pipe_y, render_setting=self.render_setting, upwards=True)
            new_pipe_down = Pipe(first_pipe_up.rect.x + current_pipe_pair_quantity * (self.pipe_distance + first_pipe_up.rect.width),
                                 pipe_y - self.pipe_gap, render_setting=self.render_setting, upwards=False)
            self.pipe_group.add(new_pipe_up)
            self.pipe_group.add(new_pipe_down)

    def update(self):
        '''
        给Group类添加一个类似于Sprite的update方法
        '''
        super().update()
        self.update_pipe_group()

    def draw(self, surface):
        super().draw(surface)
        self.pipe_group.draw(surface)
