import time
import os


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
        self.folder = './log/'
        os.makedirs(self.folder, exist_ok=True)
        self.output_file_path = self.folder + output_file_path
        self.file = open(self.output_file_path, 'a+')

    def update(self, message, level, location):
        '''
        更新自身（即输出日志）
        '''
        super().update()
        # with open(self.output_file_path, "a") as f:
        #     f.write("[{}][{}][{}] {}\n".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        #                                        level, location, message))
        self.file.write("[{}][{}][{}] {}\n".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                                                   level, location, message))


class ConsoleLoggerOberver(LoggerObserver):
    '''
    控制台日志观察者，输出日志的位置为控制台
    '''

    def __init__(self):
        super().__init__()

    def update(self, message, level, location):
        '''
        更新自身（即输出日志）
        '''
        super().update()
        print("[{}][{}][{}] {}".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                                       level, location, message))
