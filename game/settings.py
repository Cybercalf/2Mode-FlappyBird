class Setting:
    '''
    管理FlappyBird游戏的一些设置
    '''
    def __init__(self):
        self.play_sound = False
        self.show_log = False
        self.complete_render = False

    def set_preset_train(self):
        self.play_sound = False
        self.show_log = False
        self.complete_render = False

    def set_preset_play(self):
        self.play_sound = True
        self.show_log = True
        self.complete_render = True
