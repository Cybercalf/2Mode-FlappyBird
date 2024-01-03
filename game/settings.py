class Setting:
    '''
    管理FlappyBird游戏的一些设置
    '''
    # 游戏屏幕的尺寸，对应素材中背景图片的宽与高
    SCREENWIDTH, SCREENHEIGHT = 288, 512
    # 游戏帧率
    FPS = 30

    # 是否播放音效
    SOUND_PLAY = True
    # 是否在控制台打印游戏日志
    PRINT_CONSOLE_LOG = True
    # 是否显示强化学习用到的游戏界面，而非正常游戏界面
    SHOW_RL_OBSERVATION_SCREEN = False

    def __init__(self):
        pass

    def set_mode(self, mode='play'):
        '''
        切换演示环境与训练环境下不同的游戏设置
        非法的模式会按照演示模式的设置处理
        :param mode: 游戏模式，play（演示模式）/train（训练模式）
        '''
        if mode == 'train':
            self.SOUND_PLAY = False
            self.PRINT_CONSOLE_LOG = False
            self.SHOW_RL_OBSERVATION_SCREEN = True
        else:
            self.SOUND_PLAY = True
            self.PRINT_CONSOLE_LOG = True
            self.SHOW_RL_OBSERVATION_SCREEN = False
