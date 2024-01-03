import pygame
import os


class Bird(pygame.sprite.Sprite):
    '''
    小鸟
    '''

    def __init__(self, x, y, setting):
        pygame.sprite.Sprite.__init__(self)

        self.setting = setting

        # 更换小鸟图片使用的参数
        self.img_frames = ['0', '1', '2', '1']
        self.frame_idx = 0
        self.idx = 0

        # 从磁盘中加载图片文件
        self.images = {}
        for image in os.listdir('assets/sprites'):
            name, extension = os.path.splitext(image)
            if name.startswith('bird'):
                path = os.path.join('assets/sprites', image)
                self.images[name] = pygame.image.load(path)

        self.image = self.images['bird0_' + self.img_frames[self.idx]]
        # 图像矩形
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y
        # 小鸟在y轴上的速度，注意正方向为竖直向下
        # 注意速度时间单位为1帧的时间
        self.y_vel = -5.5 * 60 / self.setting.FPS
        # 重力
        self.gravity = 0.4 * ((60 / self.setting.FPS) ** 2)
        # 小鸟爬升时的各项初始参数
        # self.y_vel_after_flap = -6 * 60 / FPS
        self.y_vel_after_flap = -5.5 * 60 / self.setting.FPS

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

        # 约每1/6秒切换一次小鸟的图片
        self.frame_idx = (self.frame_idx + 1) % int(self.setting.FPS / 6)
        self.idx = (self.frame_idx % 4)
        self.image = self.images['bird0_' + self.img_frames[self.idx]]
