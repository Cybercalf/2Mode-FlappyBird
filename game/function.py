import pygame


def update(*sprites):
    '''
    调用传入的所有Sprite与Group类的update()
    '''
    for sprite in sprites:
        if isinstance(sprite, pygame.sprite.Sprite) or isinstance(
                sprite, pygame.sprite.Group):
            sprite.update()


def draw(*sprites, surface):
    '''
    调用传入的所有Sprite与Group类的draw()
    注意参数的传入顺序影响着渲染的顺序
    '''
    for sprite in sprites:
        if isinstance(sprite, pygame.sprite.Sprite) or isinstance(
                sprite, pygame.sprite.Group):
            sprite.draw(surface)


def redraw(*sprites, surface):
    '''
    将画布填满黑色，再调用传入的所有Sprite与Group类的draw()
    注意参数的传入顺序影响着渲染的顺序
    '''
    surface.fill(0)
    for sprite in sprites:
        if isinstance(sprite, pygame.sprite.Sprite) or isinstance(
                sprite, pygame.sprite.Group):
            sprite.draw(surface)
