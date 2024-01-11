import pygame
from ..sprite.background import NormalBG
from ..sprite.floor import Floor
from ..util.interface import GameSprite
from ..util import function as function


class EndWindow():
    '''
    游戏结束时展示的界面
    '''

    def __init__(self, setting):
        super().__init__()
        self.setting = setting
        self.floor = Floor(setting=self.setting)
        self.background = NormalBG()
        self.sprite_gameover = GameSprite()
        self.sprite_gameover.image = pygame.image.load('flappybird/assets/sprites/text_game_over.png')
        self.sprite_gameover.rect = self.sprite_gameover.image.get_rect()
        self.sprite_gameover.rect.x = (self.setting.SCREENWIDTH - self.sprite_gameover.image.get_width()) / 2
        self.sprite_gameover.rect.y = (self.setting.SCREENHEIGHT - self.floor.image.get_height() -
                                       self.sprite_gameover.image.get_height()) / 2

    def show(self, surface):
        '''
        展示此页面
        '''
        function.redraw(self.background, self.floor, self.sprite_gameover, surface=surface)
        pygame.display.update()
        while True:
            # 通过pygame.event模块，不断获取当前发生的事件
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    quit()
                # 按下空格时，从结束界面返回，切换到下一界面
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                    return
