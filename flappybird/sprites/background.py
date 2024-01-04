import pygame


class Background(pygame.sprite.Sprite):
    '''
    FlappyBird游戏背景
    '''

    def __init__(self):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.Surface((0, 0))
        self.x = 0
        self.y = 0

    def update(self):
        pass

    def draw(self, surface: pygame.Surface):
        '''
        将自身渲染在画面上
        :param screen: 渲染的目标画面
        '''
        surface.blit(self.image, (self.x, self.y))


class BlackBG(Background):
    '''
    游戏的黑色背景，用于生成图像传入网络
    '''

    def __init__(self):
        super().__init__()
        self.image = pygame.image.load('assets/sprites/bg_black.png')


class NormalBG(Background):
    '''
    游戏的正常背景，用于展示
    '''

    def __init__(self):
        super().__init__()
        self.image = pygame.image.load('assets/sprites/bg_day.png')
