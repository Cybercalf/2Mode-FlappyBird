import pygame
from .interface import GameElement, GameSprite


class Digit(GameSprite):
    '''
    单个数字
    '''

    def __init__(self, x, y, digit):
        super().__init__()
        self.x = x
        self.y = y
        self.image = pygame.image.load(
            'assets/sprites/number_score_0{}.png'.format(digit))


class ScoreManager(GameElement):
    '''
    管理分数变化与显示的类
    '''
    # TODO: 尝试把分数变化结合进update()方法中

    def __init__(self, setting):
        super().__init__()

        self.setting = setting

        self.score = 0

        self.digits = pygame.sprite.Group()
        for i in range(10):
            digit = Digit(0, 0, i)
            self.digits.add(digit)

    def score_increase_1(self):
        '''
        游戏分数+1
        '''
        self.score += 1

    def draw(self, surface):
        '''
        将分数图片渲染在指定的画面上
        '''
        super().draw(surface)

        score_str = str(self.score)

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
