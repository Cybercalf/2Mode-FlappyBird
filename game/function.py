import pygame


def update(*sprites):
    for sprite in sprites:
        if isinstance(sprite, pygame.sprite.Sprite) or isinstance(sprite, pygame.sprite.Group):
            sprite.update()
