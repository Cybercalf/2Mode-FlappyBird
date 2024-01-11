class InvalidActionException(Exception):
    '''
    游戏接收到无效action时产生的异常
    '''

    def __init__(self, message='Default message'):
        self.message = message

    def __str__(self) -> str:
        return 'InvalidActionException: {}'.format(self.message)


class InvalidPlayerException(Exception):
    '''
    游戏玩家无效时产生的异常
    '''

    def __init__(self, message='Default message'):
        self.message = message

    def __str__(self) -> str:
        return 'InvalidPlayerException: {}'.format(self.message)
