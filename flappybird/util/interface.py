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
    一个游戏内不继承自pygame.sprite.Sprite的元素，具有`update()`与`draw()`方法
    '''

    def __init__(self):
        super().__init__()


class GameSprite(pygame.sprite.Sprite, GameElement):
    '''
    一个游戏内的sprite，具有`update()`与`draw()`方法

    `GameSprite`的子类必须具有`self.image`和`self.rect`属性

    一个简单的子类的__init__()例子：
    ```python
    def __init__(self):
        super().__init__()
        self.image = pygame.Surface((0, 0))
        self.rect = self.image.get_rect()
    ```
    '''

    def __init__(self):
        super().__init__()
        self.image = pygame.Surface((0, 0))
        self.rect = self.image.get_rect()

    def draw(self, surface):
        super(GameElement, self).draw(surface)
        surface.blit(self.image, self.rect)


class Counter():
    '''
    一个计数器，可以用于记录游戏分数
    '''

    def __init__(self):
        self.number = 0

    def increase(self, num=1):
        '''
        改变计数
        :param num: 要增加的计数值，默认为1
        '''
        self.number += num

    def get_number(self):
        '''
        获取当前计数器的值
        '''
        return self.number

    def set(self, num):
        '''
        设置计数
        :param num: 要设置的计数值
        '''
        self.number = num

    def reset(self):
        '''
        重置计数为0
        '''
        self.set(0)
