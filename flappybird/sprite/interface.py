import pygame


class Updateable():
    '''
    一个具有update()方法的接口
    '''

    def __init__(self):
        pass

    def update(self):
        pass


class Drawable():
    '''
    一个具有draw()方法的接口
    '''

    def __init__(self):
        pass

    def draw(self, surface: pygame.Surface):
        pass


class GameElement(Drawable, Updateable):
    '''
    一个游戏内不继承自pygame.sprite.Sprite的元素，具有update()与draw()方法
    '''

    def __init__(self):
        super().__init__()


class GameSprite(pygame.sprite.Sprite, GameElement):
    '''
    一个游戏内的sprite，具有update()与draw()方法
    '''

    def __init__(self):
        super().__init__()
        self.image = pygame.Surface((0, 0))
        self.rect = self.image.get_rect()

    def draw(self, surface):
        super(GameElement, self).draw(surface)
        surface.blit(self.image, self.rect)
