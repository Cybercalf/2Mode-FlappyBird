import random

import numpy as np
import torch
import torch.nn
from torch.autograd import Variable

from .custom_enum import ExplorationMethod
from .replay import ReplayMemory


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

    def __init__(self, memory_size: int, device: torch.device):
        self.actions = 2  # 游戏动作空间（可执行的动作数量）
        self.memory = ReplayMemory(memory_size)  # 经验回放池
        self.device = device

        # agent与环境交互的时间步
        self.time_step = 0

        self.reset_state()

    def reset_state(self):
        '''
        重置目前的状态
        '''
        self.current_state = FlappyAgent.empty_state

    def update_current_state(self, new_frame):
        '''
        更新当前状态
        '''
        # 把当前state（4帧图像）最早的一帧去除，加上从游戏得到的最新的一帧图像，作为下一个state
        self.current_state = np.append(
            self.current_state[1:, :, :], new_frame.reshape((1,) + new_frame.shape),
            axis=0
        )
        return self.current_state

    def get_action_based_on_fixed_pr(self, pr_of_flapping=0.075):
        '''
        依据预设的固定概率，从动作空间中随机选择一个动作
        :param pr_of_flapping: 选择动作“拍翅膀”的概率
        '''
        action = np.zeros(self.actions, dtype=np.uint8)
        action[1 if random.random() < pr_of_flapping else 0] = 1
        return action

    def get_optim_action(self, network: torch.nn.Module):
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
        with torch.no_grad():
            state_var = Variable(torch.from_numpy(self.current_state)).unsqueeze(0).to(self.device)
        q_value = network(state_var)
        _, action_index = torch.max(q_value, dim=1)
        action_index = action_index.data[0].item()
        action = np.zeros(self.actions, dtype=np.uint8)
        action[action_index] = 1
        return action

    def get_action_based_on_exploration(self, network: torch.nn.Module,
                                        exploration_method=ExplorationMethod.EPSILON_GREEDY, epsilon=1.0, tau=0.5):
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
        """
        if exploration_method == ExplorationMethod.EPSILON_GREEDY:
            if random.random() <= epsilon:
                action = self.get_action_based_on_fixed_pr()
            else:
                action = self.get_optim_action(network)

        elif exploration_method == ExplorationMethod.BOLTZMANN_EXPLORATION:
            with torch.no_grad():
                state_var = Variable(torch.from_numpy(self.current_state)).unsqueeze(0).to(self.device)
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
            q_value = network(state_var)
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
