import random
from collections import deque
import numpy as np
import torch
import torch.nn
from torch.autograd import Variable


class FlappyBirdQNetwork(torch.nn.Module):
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
        super(FlappyBirdQNetwork, self).__init__()

        # 用于表示当前是否正在训练的标志变量
        self.train = None
        # 用于回放的“记忆”，结构是一个双端队列
        self.replay_memory = deque()
        # 初始化其他参数
        self.time_step = 0
        self.epsilon = epsilon
        self.actions = FlappyBirdQNetwork.ACTIONS
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
        # 经过第2个卷积层之后，tensor后3个维度的尺寸如map_size所示
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
        self.current_state = state if state is not None else FlappyBirdQNetwork.empty_state

    def store_transition(self, o_next, action, reward, terminal):
        '''
        存储状态转移情况
        :param o_next: 下一个观测
        :param action: agent的动作
        :param reward: agent在这一帧得到的奖励值
        :param terminal: 在这一帧游戏是否结束
        '''
        # TODO: 解释下面一句代码的具体含义，该句代码在各处被大量使用
        # 目前推测，含义为把当前state（4帧图像）最早的一帧去除，加上从游戏得到的最新的一帧图像，作为下一个state
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

    def get_action_randomly(self, pr_of_flapping=0.2):
        '''
        随机选择一个动作（action）
        '''
        # TODO: 为什么action的数据格式是np.float32?换成uint8岂不是更省空间？
        action = np.zeros(self.actions, dtype=np.float32)
        """
        设agent随机选取action时，拍翅膀的概率为p
        agent在随机选取动作时，有(1-p)的概率不拍翅膀，有p的概率拍翅膀
        原作者设定p=0.2
        之所以选择这个概率，有可能是为了使参数更快收敛，或者说，更容易找到飞得更远的方式，提高前期学习效率
        """
        action[1 if random.random() < pr_of_flapping else 0] = 1
        return action

    def get_optim_action(self):
        '''
        根据当前的状态，选择最佳的动作（action）
        '''

        """
        torch.autograd包是pytorch搭建神经网络的核心，它为张量上的所有操作提供了自动求导机制；torch.autograd.Variable类的作用是包装Tensor，将一个tensor其变成计算图中的一个节点(变量)
        一个tensor转换为Variable后，将有以下三个重要的属性：
        .data：访问这个Variable存储的张量数据，即原始的张量值
        .grad：访问这个Variable的梯度信息
        grad_fn：访问这个Variable的运算信息，表示该变量是用户创建的变量还是中间计算出的结果变量。当变量是计算图的leaf nodes（叶节点）时，.grad_fn为False，当变量是计算图中间的结果变量时，.grad_fn为True。

        q_value: 选取不同action的q值
        尺寸：torch.Size([1, 2])
        q_value中的两个数值分别代表小鸟不拍翅膀和拍翅膀的预期收益
        每次采取最优动作时，会选取数值较大（即预期收益更高）的一个动作
        eg. q_value = tensor([[15.4445,  2.2350]], device='cuda:0', grad_fn=<AddmmBackward0>)，则这一帧小鸟不拍翅膀的预期收益更高，最优选择就是不拍翅膀
        """
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

    def get_action(self, setting):
        '''
        根据当前状态（state）选择动作（action）
        :param: exploration: 探索方式，e=epsilon-greedy，b=Boltzmann Exploration
        '''

        """
        在训练过程中，有2中exploration方法：
        1.依据epsilon-greedy策略，agent有概率不按照自己学习得到的规则选择最佳动作，而是随机选择动作，概率由epsilon的值规定
        2.依据Boltzmann Exploration方法，依据不同action的Q值大小确定选择每种action的概率，action对应的Q值越大，选择它的概率越高

        如果不是训练过程，agent一定会选择（它自己认为的）最佳动作

        注意：如果exploration策略从一开始就采用Boltzmann Exploration，训练的收敛速度会很慢很慢
        推测原因为，由于游戏奖励值设的比较小，导致从qnetwork中得出的两个action值对应的Q值相差不大，套上exp之后差距就更小了
        这导致每次小鸟用Boltzmann Exploration选择action时，选取两个动作的概率很接近
        Boltzmann Exploration还有一个控制选择动作概率的参数，
        参数过小，选择action更加贪心，前期训练可能收敛更快，但后期网络几乎不会再作探索
        参数过大，选择action更随机，前期训练收敛速度很慢

        TODO: Boltzmann Exploration引入了大量的浮点数运算，尝试加速

        TODO: 优化三种选择action的方法的逻辑，去掉重复代码
        """
        if setting.exploration_method == 'Epsilon Greedy':
            # Epsilon Greedy
            if self.train and random.random() <= self.epsilon:
                action = self.get_action_randomly()
            else:
                action = self.get_optim_action()
        elif setting.exploration_method == 'Boltzmann Exploration':
            # Boltzmann Exploration
            state = self.current_state
            with torch.no_grad():
                state_var = Variable(torch.from_numpy(state)).unsqueeze(0)
            if self.use_cuda:
                state_var = state_var.cuda()
            """
            eg. q_value = tensor([[1.0, 2.0]]), tau = 0.5
            probability = exp(1.0/0.5) / (exp(1.0/0.5) + exp(2.0/0.5))
            对probability的计算还有一种写法：
            probability = torch.nn.functional.normalize(
                torch.exp(q_value / setting.boltzmann_exploration.tau),
                p=1
            )[0][0].item()
            实测这种方法运行速度不如自己求比例
            下面的算法是目前想到的算法中，经测试最快的
            """
            q_value = self.forward(state_var)
            q_value_after_control = torch.exp(q_value / setting.boltzmann_exploration.tau)
            probability = (q_value_after_control / torch.sum(q_value_after_control))[0][0].item()
            action = np.zeros(self.actions, dtype=np.float32)
            action[0 if random.random() < probability else 1] = 1
            # TODO: debug
            print("q_value: {}".format(q_value))
            # print("q_value_after_control_by_tau: {}".format(q_value_after_control_by_tau))
            # print("q_value_after_control_by_tau_exp: {}".format(q_value_after_control_by_tau_exp))
            print("probability: {}".format(probability))
        else:
            raise ValueError('invalid exploration method when getting action')

        return action

    def increase_time_step(self, time_step=1):
        '''
        增加时间步的值。
        时间步（time step）在程序中（或者说在我们构建的系统中）是时间的一个度量单位，用于将时间范围分割成连续的小时间间隔。
        '''
        self.time_step += time_step
