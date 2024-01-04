class LoggerObserver():
    '''
    日志观察者
    '''

    def __init__(self):
        pass

    def update(self):
        pass


class FileLoggerObserver(LoggerObserver):
    '''
    文件日志观察者，输出日志的位置为文件
    '''

    def __init__(self, output_file_path):
        super().__init__()
        self.output_file_path = output_file_path

    def update(self, message, location):
        '''
        更新自身（即输出日志）
        '''
        super().update()
        with open(self.output_file_path, "a") as f:
            f.write("[{}] {}".format(location, message))


class ConsoleLoggerOberver(LoggerObserver):
    '''
    控制台日志观察者，输出日志的位置为控制台
    '''

    def __init__(self):
        super().__init__()

    def update(self, message, location):
        '''
        更新自身（即输出日志）
        '''
        super().update()
        print("[{}] {}".format(location, message))
