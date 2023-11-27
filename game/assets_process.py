import pygame
import os


def load_assets():
    '''
    从磁盘中加载游戏所需的图片与音效文件
    '''
    # 从磁盘中加载图片文件
    BIRD_IMAGES = {}
    PIPE_IMAGES = {}
    SCORE_IMAGES = {}
    for image in os.listdir('assets/sprites'):
        name, extension = os.path.splitext(image)
        if name.startswith('bird'):
            path = os.path.join('assets/sprites', image)
            BIRD_IMAGES[name] = pygame.image.load(path)
        if name.startswith('pipe'):
            path = os.path.join('assets/sprites', image)
            PIPE_IMAGES[name] = pygame.image.load(path)
        if name.startswith('number_score'):
            path = os.path.join('assets/sprites', image)
            SCORE_IMAGES[name] = pygame.image.load(path)

    IMAGES = {}
    # 为了提高卷积神经网络的精度，将背景图片变为纯黑色以减小干扰
    # IMAGES['bgpic'] = pygame.image.load('assets/sprites/bg_day.png')
    IMAGES['bgpic'] = pygame.image.load('assets/sprites/bg_black.png')
    IMAGES['floor'] = pygame.image.load('assets/sprites/land.png')
    IMAGES['LIST_BIRD'] = BIRD_IMAGES
    IMAGES['LIST_PIPE'] = PIPE_IMAGES
    IMAGES['LIST_SCORE'] = SCORE_IMAGES

    # 从磁盘中加载音效文件
    SOUNDS = {}
    SOUNDS['background'] = pygame.mixer.Sound('assets/audio/background.wav')
    SOUNDS['die'] = pygame.mixer.Sound('assets/audio/gameover.ogg')
    SOUNDS['hit'] = pygame.mixer.Sound('assets/audio/hit.ogg')
    SOUNDS['score'] = pygame.mixer.Sound('assets/audio/score.ogg')
    SOUNDS['flap'] = pygame.mixer.Sound('assets/audio/jump.ogg')

    return IMAGES, SOUNDS
