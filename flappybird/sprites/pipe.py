import pygame
import random


class Pipe(pygame.sprite.Sprite):
    '''
    水管
    '''

    def __init__(self, x, y, setting, upwards=True):
        pygame.sprite.Sprite.__init__(self)
        if upwards:
            self.image = pygame.image.load('assets/sprites/pipe_up.png')
            self.rect = self.image.get_rect()
            self.rect.x = x
            self.rect.top = y
        else:
            self.image = pygame.image.load('assets/sprites/pipe_down.png')
            self.rect = self.image.get_rect()
            self.rect.x = x
            self.rect.bottom = y
        # 水平速度，向右为正方向
        # 注意速度时间单位为1帧的时间
        self.x_vel = -3 * 60 / setting.FPS

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
    pipe_distance = 220
    # 上下水管的间距
    pipe_gap = 130

    def __init__(self, *sprites, setting):
        super().__init__(*sprites)

        self.setting = setting

        # 生成水管实例，一次生成一对水管
        self.pipe_group = pygame.sprite.Group()
        for i in range(self.pipe_quantity):
            pipe_y = random.randint(
                int(self.setting.SCREENHEIGHT * 0.3), int(self.setting.SCREENHEIGHT * 0.7))
            new_pipe_up = Pipe(self.setting.SCREENWIDTH + i *
                               self.pipe_distance, pipe_y, setting=self.setting, upwards=True)
            new_pipe_down = Pipe(self.setting.SCREENWIDTH + i * self.pipe_distance, pipe_y -
                                 self.pipe_gap, setting=self.setting, upwards=False)
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
        1.移动每一根水管的位置（调用每一个Pipe的update()方法）
        2.如果有水管从左边移出屏幕，把它从列表中删除，在右侧新添一个水管
        '''
        # 对整个Group使用update()方法，会触发Group内每一个Sprite（水管）的update()方法，水管更改位置的算法写在Pipe类的update()里面，很简单
        self.pipe_group.update()
        # 如果有水管从左边移出屏幕，把它从列表中删除，在右侧新添一个水管
        first_pipe_up = self.get_first_pipe_up()
        first_pipe_down = self.get_first_pipe_down()
        if first_pipe_up.rect.right < 0:
            pipe_y = random.randint(
                int(self.setting.SCREENHEIGHT * 0.3), int(self.setting.SCREENHEIGHT * 0.7))
            new_pipe_up = Pipe(
                first_pipe_up.rect.x + self.pipe_quantity * self.pipe_distance, pipe_y, setting=self.setting, upwards=True)
            new_pipe_down = Pipe(first_pipe_up.rect.x + self.pipe_quantity *
                                 self.pipe_distance, pipe_y - self.pipe_gap, setting=self.setting, upwards=False)
            self.pipe_group.add(new_pipe_up)
            self.pipe_group.add(new_pipe_down)
            first_pipe_up.kill()
            first_pipe_down.kill()
        pass

    def update(self):
        '''
        给Group类添加一个类似于Sprite的update方法
        '''
        self.update_pipe_group()

    def draw(self, canvas):
        self.pipe_group.draw(canvas)
