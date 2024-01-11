import pygame


class SoundManager:
    '''
    管理游戏音效
    '''
    def __init__(self):
        self.sounds = self.load_sounds()

    def load_sounds(self):
        '''
        从磁盘中加载游戏所需的音效文件
        '''
        SOUNDS = {}
        SOUNDS['background'] = pygame.mixer.Sound('flappybird/assets/audio/background.wav')
        SOUNDS['die'] = pygame.mixer.Sound('flappybird/assets/audio/gameover.ogg')
        SOUNDS['hit'] = pygame.mixer.Sound('flappybird/assets/audio/hit.ogg')
        SOUNDS['score'] = pygame.mixer.Sound('flappybird/assets/audio/score.ogg')
        SOUNDS['flap'] = pygame.mixer.Sound('flappybird/assets/audio/jump.ogg')
        return SOUNDS

    def play(self, key, volume=1, loop=False):
        '''
        播放指定音乐
        '''
        self.sounds[key].set_volume(volume)
        self.sounds[key].play(loops=-1 if loop else 0)
