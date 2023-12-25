import random
from collections import deque
import numpy as np
import torch
import torch.nn
from torch.autograd import Variable


class BrainDQN(torch.nn.Module):
    '''
    依据xmfbit大神设计的DQN改造
    '''

    # 规定了agent可用的action数量
    # 对应FlappyBird：拍翅膀和不拍翅膀
    ACTIONS = 2

    """
    初始化一个数组，尺寸128*72，数据类型np.float32
    初始化一个“状态”，由四个数组在axis=0处叠加得到
    推测empty_frame代表一帧画面，empty_state代表由连续的4帧组成的一个状态
    """
    empty_frame = np.zeros((128, 72), dtype=np.float32)
    empty_state = np.stack(
        (empty_frame, empty_frame, empty_frame, empty_frame),
        axis=0
    )

    def __init__(self, epsilon, mem_size, cuda):
        '''
        创建实例的方法

        :param epsilon: 用于exploration的值，控制agent随机选择action的概率
        :param mem_size: 经验回放（experience replay）用到的内存大小
        :param cuda: 是否使用cuda
        '''
        super(BrainDQN, self).__init__()

        # 用于表示当前是否正在训练的标志变量
        self.train = None
        # 用于回放的“记忆”，结构是一个双端队列
        self.replay_memory = deque()
        # 初始化其他参数
        self.time_step = 0
        self.epsilon = epsilon
        self.actions = self.ACTIONS
        self.mem_size = mem_size
        self.use_cuda = cuda
        # 创建神经网络
        self.createQNetwork()

    def createQNetwork(self):
        '''
        创建神经网络，在创建类实例时会自动调用
        模型结构：卷积层-卷积层-全连接层-全连接层
        '''

        """
        torch.nn.Conv2d()
        in_channels: 第一个参数，输入的通道数
        out_channels: 第二个参数，输出的通道数
        kernel_size: 卷积核的大小，一般采用长宽相同的卷积核，所以kernel_size=8代表一个8*8的卷积核
        stride: 卷积核在图像窗口上平移的步长
        padding: 在图像四周（用0）填充多大的尺寸。例：原图像32*32，padding=1，填充后的图像尺寸34*34

        torch.nn.ReLU()
        inplace: 计算的值是否覆盖输入的值。若inplace=True，计算后输入的原始变量会被新值覆盖，这样做能省内存
        """
        self.conv1 = torch.nn.Conv2d(4, 32, kernel_size=8, stride=4, padding=2)
        self.relu1 = torch.nn.ReLU(inplace=True)
        self.conv2 = torch.nn.Conv2d(
            32, 64, kernel_size=4, stride=2, padding=1)
        self.relu2 = torch.nn.ReLU(inplace=True)
        # TODO: 搞懂map_size含义
        self.map_size = (64, 16, 9)

        """
        torch.nn.Linear()
        线性层（全连接层），将输入与权重矩阵相乘并加上偏置，然后通过激活函数进行非线性变换
        in_features: 第一个参数。输入的神经元个数
        out_features: 第二个参数。输出的神经元个数
        """
        self.fc1 = torch.nn.Linear(
            self.map_size[0] * self.map_size[1] * self.map_size[2], 256
        )
        self.relu3 = torch.nn.ReLU(inplace=True)
        self.fc2 = torch.nn.Linear(256, self.actions)

    def get_q_value(self, o):
        '''
        根据当前的观测o，给出对应的Q值
        '''

        # TODO: 搞清楚各个维度上数据的含义，尤其是进入网络之前的数据，以及经过第一个卷积层后数据尺寸是怎么变化的
        """
        before go into the net, size: torch.Size([1, 4, 128, 72])
        after self.conv1, size: torch.Size([1, 32, 32, 18])
        after self.relu1, size: torch.Size([1, 32, 32, 18])
        after self.conv2, size: torch.Size([1, 64, 16, 9])
        after self.relu2, size: torch.Size([1, 64, 16, 9])
        after out.view(out.size()[0], -1), size: torch.Size([1, 9216])
        after self.fc1, size: torch.Size([1, 256])
        after self.relu3, size: torch.Size([1, 256])
        after self.fc2, size: torch.Size([1, 2])
        """
        out = self.conv1(o)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        """
        用torch.view()改变tensor尺寸，不改变数据
        改变前后各个维度上尺寸的乘积相等（数据量相等）
        -1 代表这个维度的尺寸由电脑自动计算
        注意改变尺寸前后的tensor共享同一块内存
        """
        out = out.view(out.size()[0], -1)
        out = self.fc1(out)
        out = self.relu3(out)
        out = self.fc2(out)
        return out

    def forward(self, o):
        '''
        用观测o得到Q值，在此过程中得到MSE损失函数
        '''
        q = self.get_q_value(o)
        return q

    def set_train(self):
        '''
        进入训练阶段（phase TRAIN）
        '''
        self.train = True

    def set_eval(self):
        '''
        进入评估阶段（phase EVALUATION）
        '''
        self.train = False

    def set_initial_state(self, state=None):
        '''
        设置原始状态
        如果不指定原始状态，就用默认的empty_state
        '''
        self.current_state = state if state is not None else BrainDQN.empty_state

    def store_transition(self, o_next, action, reward, terminal):
        '''
        存储状态转移情况
        :param o_next: 下一个观测
        :param action: agent的动作
        :param reward: agent在这一帧得到的奖励值
        :param terminal: 在这一帧游戏是否结束
        '''
        next_state = np.append(
            self.current_state[1:, :, :], o_next.reshape((1,) + o_next.shape),
            axis=0
        )
        self.replay_memory.append(
            (self.current_state, action, reward, next_state, terminal)
        )
        if len(self.replay_memory) > self.mem_size:
            self.replay_memory.popleft()
        """
        如果游戏没有结束(not terminal)，继续更新状态信息，否则将当前状态重置为初始值
        """
        if not terminal:
            self.current_state = next_state
        else:
            self.set_initial_state()

    def get_action_randomly(self):
        '''
        随机选择一个动作（action）
        '''
        action = np.zeros(self.actions, dtype=np.float32)
        """
        原作者把随机选择动作的概率写死在代码里
        agent在随机选取动作时，有0.8的概率不拍翅膀，有0.2的概率拍翅膀
        之所以选择这个概率，有可能是为了使参数更快收敛，或者说，更容易找到飞得更远的方式，提高前期学习效率
        TODO: 让随机选取不同动作的概率不再硬编码
        """
        action[0 if random.random() < 0.8 else 1] = 1
        return action

    def get_optim_action(self):
        '''
        根据当前的状态，选择最佳的动作（action）
        '''

        # TODO: 看懂get_optim_action的具体逻辑
        state = self.current_state
        with torch.no_grad():
            state_var = Variable(torch.from_numpy(state)).unsqueeze(0)
        if self.use_cuda:
            state_var = state_var.cuda()
        q_value = self.forward(state_var)
        _, action_index = torch.max(q_value, dim=1)
        action_index = action_index.data[0].item()
        action = np.zeros(self.actions, dtype=np.float32)
        action[action_index] = 1
        return action

    def get_action(self):
        '''
        根据当前状态（state）选择动作（action）
        '''

        """
        在训练过程中，依据epsilon-greedy策略，agent有概率不按照自己学习得到的规则选择最佳动作，而是随机选择动作，概率由epsilon的值规定
        如果不是训练过程，agent一定会选择（它自己认为的）最佳动作
        """
        if self.train and random.random() <= self.epsilon:
            return self.get_action_randomly()
        return self.get_optim_action()

    def increase_time_step(self, time_step=1):
        '''
        增加时间步的值。
        时间步（time step）在程序中（或者说在我们构建的系统中）是时间的一个度量单位，用于将时间范围分割成连续的小时间间隔。
        '''
        self.time_step += time_step
