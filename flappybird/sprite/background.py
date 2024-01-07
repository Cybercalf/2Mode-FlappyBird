import pygame
from .interface import DrawableSprite


class Background(DrawableSprite):
    '''
    FlappyBird游戏背景
    '''

    def __init__(self):
        super().__init__()


class BlackBG(Background):
    '''
    游戏的黑色背景，用于生成图像传入网络
    '''

    def __init__(self):
        super().__init__()
        self.image = pygame.image.load('assets/sprites/bg_black.png')
        self.rect = self.image.get_rect()


class NormalBG(Background):
    '''
    游戏的正常背景，用于展示
    '''

    def __init__(self):
        super().__init__()
        self.image = pygame.image.load('assets/sprites/bg_day.png')
        self.rect = self.image.get_rect()
