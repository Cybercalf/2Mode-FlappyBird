import pygame


class Drawable():
    '''
    一个具有draw()方法的接口
    '''

    def __init__(self):
        pass

    def draw(self, surface):
        pass


class DrawableSprite(pygame.sprite.Sprite, Drawable):
    '''
    一个具有draw()方法的sprite
    '''

    def __init__(self):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.Surface((0, 0))
        self.rect = self.image.get_rect()

    def draw(self, surface: pygame.Surface):
        super().draw(surface)
        surface.blit(self.image, self.rect)
