# TODO: 让窗口类引用floor类
import pygame


class EndWindow():
    def __init__(self, setting):
        self.setting = setting
        self.images = {}
        self.images['gameover'] = pygame.image.load(
            'assets/sprites/text_game_over.png')
        self.images['bgpic'] = pygame.image.load('assets/sprites/bg_day.png')
        self.images['floor'] = pygame.image.load('assets/sprites/land.png')
        self.gameover_x = (self.setting.SCREENWIDTH -
                           self.images['gameover'].get_width()) / 2
        self.gameover_y = (
            self.setting.SCREENHEIGHT - self.images['floor'].get_height() - self.images['gameover'].get_height()) / 2

    def show(self, surface):
        '''
        展示此页面
        '''
        surface.blit(self.images['bgpic'], (0, 0))
        surface.blit(self.images['floor'],
                     (0, self.setting.SCREENHEIGHT - self.images['floor'].get_height()))
        surface.blit(self.images['gameover'],
                     (self.gameover_x, self.gameover_y))
        pygame.display.update()
        while True:
            # 通过pygame.event模块，不断获取当前发生的事件
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    quit()
                # 按下空格时，从结束界面返回，切换到下一界面
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                    return
