class LoggerSubject():
    '''
    日志主题类
    当主题状态发生变化时，将通知所有订阅主题的日志观察者，以输出日志
    '''

    def __init__(self):
        # 观察者列表，类型为字典，键值对为"日志级别:该级别下的观察者列表"
        self.observers = {}

    def register_observer(self, observer, level):
        '''
        注册观察者
        :param observer: 观察者类
        :param level: 要注册的日志级别
        '''
        if level not in self.observers:
            self.observers[level] = []
        self.observers[level].append(observer)

    def remove_observer(self, observer, level):
        if level in self.observers:
            self.observers[level].remove(observer)

    def notify_observers(self, message, level, location=''):
        '''
        通知所有指定级别的观察者更新自身
        :param message: 传输的信息
        :param level: 日志级别
        :param location: 日志产生的代码位置（可用于调试）
        '''
        if level in self.observers:
            for observer in self.observers[level]:
                observer.update(message, level, location)

    # 为notify_observers()方法起别名
    generate_log = notify_observers
