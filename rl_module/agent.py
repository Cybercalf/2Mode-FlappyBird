import random
from collections import deque

import numpy as np
import torch
import torch.nn
from torch.autograd import Variable

from .nn import FlappyQNet, FlappyVAQNet
from .custom_enum import NetStruct, ExplorationMethod


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


class FlappyAgent():
    '''
    管理QNetwork、与环境交互、存储经验
    '''

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

    def __init__(self, memory_size: int, net_struct: NetStruct, use_cuda: bool):
        self.actions = 2  # 游戏动作空间（可执行的动作数量）
        self.memory = ReplayMemory(memory_size)  # 经验回放池

        self.net_struct = net_struct
        # flappy_qnet = FlappyQNet()
        # flappy_vaqnet = FlappyVAQNet()
        # self.net = flappy_vaqnet if self.net_struct == NetStruct.DUELING else flappy_qnet  # 神经网络
        self.net = FlappyVAQNet() if self.net_struct == NetStruct.DUELING else FlappyQNet()  # 神经网络
        self.device = torch.device('cuda:0') if use_cuda else torch.device('cpu')  # 计算使用的设备
        # 将神经网络转移到指定的设备上计算
        self.net.to(self.device)

        # agent与环境交互的时间步
        self.time_step = 0

    def get_q_value(self, observation):
        '''
        根据当前的观测o，给出对应的Q值
        :param observation: 观测的状态。size: torch.Size([1, 4, 128, 72])
        :return 动作空间内每个动作的Q值。size: torch.Size([1, 2])
        '''
        return self.net(observation)

    def init_state(self, state=None):
        '''
        初始化当前状态
        若不指定状态值，则用默认值初始化
        '''
        self.current_state = FlappyAgent.empty_state if state is None else state

    def get_action_based_on_fixed_pr(self, pr_of_flapping=0.075):
        '''
        依据预设的固定概率，从动作空间中随机选择一个动作
        :param pr_of_flapping: 选择动作“拍翅膀”的概率
        '''
        action = np.zeros(self.actions, dtype=np.uint8)
        action[1 if random.random() < pr_of_flapping else 0] = 1
        return action

    def get_optim_action(self):
        '''
        依据神经网络给出的Q值，选取后续奖励期望最大的动作
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
            state_var = Variable(torch.from_numpy(state)).unsqueeze(0).to(self.device)
        q_value = self.net(state_var)
        _, action_index = torch.max(q_value, dim=1)
        action_index = action_index.data[0].item()
        action = np.zeros(self.actions, dtype=np.uint8)
        action[action_index] = 1
        return action

    def get_action_based_on_exploration(self, exploration_method=ExplorationMethod.EPSILON_GREEDY, epsilon=1.0, tau=0.5):
        '''
        依据exploration选择动作
        :param exploration_method: 探索的方式。默认为epsilon-greedy。
        :param epsilon: epsilon-greedy策略使用的参数，代表agent随机选择动作的概率
        :param tau: Boltzmann Exploration策略使用的参数，用于控制agent选择不同动作的概率差异大小
        '''
        """
        在训练过程中，有2中exploration方法：
        1.依据epsilon-greedy策略，agent有概率不按照自己学习得到的规则选择最佳动作，而是随机选择动作，概率由epsilon的值规定
        2.依据Boltzmann Exploration方法，依据不同action的Q值大小确定选择每种action的概率，action对应的Q值越大，选择它的概率越高

        注意：如果exploration策略从一开始就采用Boltzmann Exploration，训练的收敛速度会很慢很慢
        推测原因为，由于游戏奖励值设的比较小，导致从qnetwork中得出的两个action值对应的Q值相差不大，套上exp之后差距就更小了
        这导致每次小鸟用Boltzmann Exploration选择action时，选取两个动作的概率很接近
        Boltzmann Exploration还有一个控制选择动作概率的参数，
        参数过小，选择action更加贪心，前期训练可能收敛更快，但后期网络几乎不会再作探索
        参数过大，选择action更随机，前期训练收敛速度很慢

        TODO:
        1.Boltzmann Exploration引入了大量的浮点数运算，尝试加速
        2.经测试，Boltzmann Exploration用于finetune的效果也不好，找原因
        """
        if exploration_method == ExplorationMethod.EPSILON_GREEDY:
            if random.random() <= epsilon:
                action = self.get_action_based_on_fixed_pr()
            else:
                action = self.get_optim_action()

        elif exploration_method == ExplorationMethod.BOLTZMANN_EXPLORATION:
            state = self.current_state
            with torch.no_grad():
                state_var = Variable(torch.from_numpy(state)).unsqueeze(0).to(self.device)
            """
            eg. q_value = tensor([[1.0, 2.0]]), tau = 0.5
            probability = exp(1.0/0.5) / (exp(1.0/0.5) + exp(2.0/0.5))
            对probability的计算还有一种写法：
            probability = torch.nn.functional.normalize(
                torch.exp(q_value / tau),
                p=1
            )[0][0].item()
            实测这种方法运行速度不如自己求比例
            下面的算法是目前想到的算法中，经测试最快的
            """
            q_value = self.net(state_var)
            q_value_after_control = torch.exp(q_value / tau)
            probability = (q_value_after_control / torch.sum(q_value_after_control))[0][0].item()
            action = np.zeros(self.actions, dtype=np.float32)
            action[0 if random.random() < probability else 1] = 1

        else:
            raise ValueError('invalid exploration method when getting action')

        return action

    def increase_time_step(self, time_step=1):
        '''
        增加时间步的值。
        时间步（time step）在程序中（或者说在我们构建的系统中）是时间的一个度量单位，用于将时间范围分割成连续的小时间间隔。
        :param time_step: 增加的时间步。默认为1。
        '''
        self.time_step += time_step

    def store_transition(self, observation_next, action, reward, terminal):
        '''
        存储当前的状态转移过程
        :param observation_next: 下一个观测
        :param action: agent的动作
        :param reward: agent在这一帧得到的奖励值
        :param terminal: 在这一帧游戏是否结束
        '''
        # 把当前state（4帧图像）最早的一帧去除，加上从游戏得到的最新的一帧图像，作为下一个state
        next_state = np.append(
            self.current_state[1:, :, :], observation_next.reshape((1,) + observation_next.shape),
            axis=0
        )
        # 向回放池中加入当前状态转移的记录
        self.memory.push(self.current_state, action, reward, next_state, terminal)
        # 如果游戏结束(terminal)，将当前状态重置为初始值；否则，更新当前的状态信息
        if terminal:
            self.init_state()
        else:
            self.current_state = next_state
