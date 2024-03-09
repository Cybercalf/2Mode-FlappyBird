import random
from collections import deque


class ReplayMemory():
    '''
    经验回放池，存储Agent与环境交互时发生的状态转移过程，用于训练
    '''

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, current_state, action, reward, next_state, terminal):
        '''
        向回放池中添加一条状态转移记录
        :param current_state: 当前状态
        :param action: 选择的动作
        :param reward: 得到的奖励
        :param next_state: 下一状态
        :param terminal: 游戏是否结束
        '''
        # 因为deque初始化时指定了maxlen，所以在回放池装满的情况下添加新回放记录时，最旧的记录会自动删除
        self.memory.append((current_state, action, reward, next_state, terminal))

    def sample(self, batch_size):
        '''
        从回放池中随机取出一部分数据
        '''
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
