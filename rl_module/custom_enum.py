from enum import Enum


class NetStruct(Enum):
    '''
    指定网络结构的枚举类
    '''
    NORMAL = 1
    DUELING = 2


class ExplorationMethod(Enum):
    '''
    指定Exploration方法的枚举类
    '''
    EPSILON_GREEDY = 1
    BOLTZMANN_EXPLORATION = 2
