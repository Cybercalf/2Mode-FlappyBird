import pygame
from ..util.interface import GameElement, GameSprite


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


class ScoreManager(GameElement):
    '''
    管理分数变化与显示的类
    '''

    def __init__(self, setting):
        super().__init__()

        self.setting = setting

        self.score = 0

        self.digits = pygame.sprite.Group()
        for i in range(10):
            digit = Digit(0, 0, i)
            self.digits.add(digit)

    def update_score(self, step=1):
        '''
        更新游戏分数
        :param step: 游戏分数的增加量，默认为1
        '''
        self.score += 1

    def reset_score(self):
        '''
        游戏分数归0
        '''
        self.score = 0
    
    def get_score(self):
        '''
        返回游戏分数
        '''
        return self.score

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
