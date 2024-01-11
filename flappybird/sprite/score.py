import pygame
from ..util.interface import GameElement, GameSprite, Counter


class Digit(GameSprite):
    '''
    单个数字
    '''

    def __init__(self, x, y, digit):
        super().__init__()
        self.x = x
        self.y = y
        self.image = pygame.image.load(
            'flappybird/assets/sprites/number_score_0{}.png'.format(digit))


class ScoreManager(GameElement, Counter):
    '''
    管理分数变化与显示的类
    '''

    def __init__(self, setting):
        Counter.__init__(self)

        self.setting = setting

        self.digits = pygame.sprite.Group()
        for i in range(10):
            digit = Digit(0, 0, i)
            self.digits.add(digit)

    def draw(self, surface):
        '''
        将分数图片渲染在指定的画面上
        '''
        super().draw(surface)

        score_str = str(self.get_score())

        # 确定绘制位置
        n = len(score_str)
        w = self.digits.sprites()[0].image.get_width() * 1.1
        x = (self.setting.SCREENWIDTH - n * w) / 2
        y = self.setting.SCREENHEIGHT * 0.1

        # 按照游戏分数位数从高到低依次绘制图片
        for digit_str in score_str:
            digit = eval(digit_str)
            image = self.digits.sprites()[digit].image
            surface.blit(image, (x, y))
            x += w
