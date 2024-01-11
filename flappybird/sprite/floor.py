import pygame
from ..util.interface import GameSprite


class Floor(GameSprite):
    '''
    地板
    '''

    def __init__(self, setting):
        super().__init__()

        self.setting = setting

        # 地板图片
        self.image = pygame.image.load('flappybird/assets/sprites/land.png')

        self.rect = self.image.get_rect()
        # 地板在水平方向上的位置
        self.rect.x = 0
        # 计算地板图片的y值：窗口大小 - 地板图片自身的高度
        self.rect.y = self.setting.SCREENHEIGHT - self.get_height()
        # 地板的水平速度，向右为正方向
        # 注意速度时间单位为1帧的时间
        self.x_vel = -3 * 60 / self.setting.FPS
        # self.x_vel = 0

    def update(self):
        '''
        定义地板更新自身状态的方法
        '''
        super().update()
        # 计算地板图片与背景图片的宽度之差
        # 地板x坐标左移，若左移到极限则复位
        floor_gap = self.get_width() - self.setting.SCREENWIDTH
        self.rect.x += self.x_vel
        if self.rect.x <= -floor_gap:
            self.rect.x = 0

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

    def get_still_floor(self):
        '''
        获取一个x为0（没有移动过）的地板类
        '''
        return Floor(setting=self.setting)
