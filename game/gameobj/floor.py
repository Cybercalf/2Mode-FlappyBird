import pygame


class Floor(pygame.sprite.Sprite):
    '''
    地板
    '''

    def __init__(self, setting):
        pygame.sprite.Sprite.__init__(self)

        self.setting = setting

        # 地板图片
        self.image = pygame.image.load('assets/sprites/land.png')
        # 地板在水平方向上的位置
        self.x = 0
        # 计算地板图片的y值：窗口大小 - 地板图片自身的高度
        self.y = self.setting.SCREENHEIGHT - self.get_height()
        # 地板的水平速度，向右为正方向
        # 注意速度时间单位为1帧的时间
        self.x_vel = -3 * 60 / self.setting.FPS
        # self.x_vel = 0

    def update(self):
        '''
        定义地板更新自身状态的方法
        '''
        # 计算地板图片与背景图片的宽度之差
        # 地板x坐标左移，若左移到极限则复位
        floor_gap = self.get_width() - self.setting.SCREENWIDTH
        self.x += self.x_vel
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
