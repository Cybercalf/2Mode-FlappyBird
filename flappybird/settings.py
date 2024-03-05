class Setting:
    '''
    管理FlappyBird游戏的一些设置
    '''
    # 游戏屏幕的尺寸，对应素材中背景图片的宽与高
    SCREENWIDTH, SCREENHEIGHT = 288, 512
    """
    游戏帧率
    注意：所有自定义Sprite在更新自身位置时都用到了此变量，以达到游戏中物体的运动速度与现实世界的时间挂钩的效果
    （即，在游玩模式下，改变FPS不会改变游戏各物体在人眼中的运动速度，与市场上的游戏对FPS的理解一致）
    但是，因为训练模式下模型需要调用游戏连续几帧的图像，而游戏渲染帧的时间单位为一个时间步（与现实世界的时间无关），
    所以改变此值会影响到传递给模型的observation
    总结：目前人类游玩模式下，FPS可任意更改；模型训练与模型游玩时的FPS应保持一致
    """
    FPS = 30

    # 是否播放音效
    SOUND_PLAY = True
    # 是否在控制台打印游戏日志
    PRINT_CONSOLE_LOG = True
    # 是否显示强化学习用到的游戏界面，而非正常游戏界面
    SHOW_RL_OBSERVATION_SCREEN = False
    # 是否解除对游戏窗口刷新速度的控制(用于加快训练)
    UNLIMIT_SCREEN_UPDATE = False
    # 是否隐藏游戏窗口
    # TODO: 如果隐藏游戏窗口，“显示强化学习用到的游戏界面”功能就用不到了
    HIDE_WINDOW = False

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
            self.UNLIMIT_SCREEN_UPDATE = True
            self.HIDE_WINDOW = True
        else:
            self.SOUND_PLAY = True
            self.PRINT_CONSOLE_LOG = True
            self.SHOW_RL_OBSERVATION_SCREEN = False
            self.UNLIMIT_SCREEN_UPDATE = False
            self.HIDE_WINDOW = False
