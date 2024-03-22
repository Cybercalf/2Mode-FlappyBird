import sys
import os
import time
import copy

import numpy as np
import PIL.Image
from torch.autograd import Variable
import torch
import torch.nn
import torch.optim

from rl_module.agent import FlappyAgent
from rl_module.custom_enum import NetStruct, ExplorationMethod
from rl_module.file import FileHandler
from rl_module.nn import FlappyDuelingQNet, FlappyQNet
from rl_module.replay import ReplayMemory
from logger.subject import LoggerSubject
import flappybird.settings
from flappybird.game_manager import GameManager as FlappyBirdGameManager


class ProgramManager(LoggerSubject):
    '''
    程序管理器，负责运行游戏、训练模型等一系列任务
    '''

    def __init__(self):
        super().__init__()

        # 初始化文件处理工具，用于将模型信息保存到磁盘、从磁盘加载模型信息
        checkpoint_save_path = './runtime_output/checkpoint/checkpoint_' + time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()) + '/'
        self.file_handler = FileHandler(checkpoint_save_path)

    def frame_preprocess(self, frame, image_size_after_resize=(72, 128)):
        '''
        对输入的帧图像做预处理

        输入图像：512*288，rgb彩色图像

        预处理步骤：

        1.降采样，使图像尺寸变为128*72

        2.将图像转换为灰度图像

        3.将图像每个像素的灰度值映射为0或1
        '''
        # TODO: 解释为什么这里的降采样尺寸是(72, 128)，而创建空Tensor时采用(128, 72)
        resize_frame = PIL.Image.fromarray(frame).resize(image_size_after_resize).convert(mode='L')
        out = np.asarray(resize_frame).astype(np.float32)
        out[out <= 1.] = 0.0
        out[out > 1.] = 1.0
        return out

    def load_training_setting(self, setting):
        '''
        加载训练设置
        '''
        self.training_setting = setting
        self.device = torch.device('cuda:0') if self.training_setting.cuda else torch.device('cpu')

    def train(self, setting):
        '''
        加载训练设置并开始训练
        '''
        self.load_training_setting(setting)
        self.dqn_training_process()

    def dqn_training_process(self):
        '''
        采用DQN理论进行训练的过程
        '''

        # 记录模型的最好游玩效果，用time_step量化
        best_time_step = 0.

        # 初始化经验回放池
        self.replay_memory = ReplayMemory(self.training_setting.memory_size)

        """
        初始化Agent，用于选择动作
        """
        agent = FlappyAgent(
            memory_size=self.training_setting.memory_size,
            device=self.device)

        """
        检查训练过程是否基于一个给定的模型开始
        如果是，从磁盘加载网络参数
        否，按照训练设置创建新网络
        """
        if self.training_setting.resume:
            self.generate_log(message='load previous model weight: {}'.format(self.training_setting.model_path),
                              level='info', location=os.path.split(__file__)[1])
            try:
                checkpoint = self.file_handler.load(self.training_setting.model_path)
            except BaseException as e:
                self.generate_log(message='Error raised when loading model. Type: {}, Description: {}'.format(type(e), e),
                                  level='error', location=os.path.split(__file__)[1])
                sys.exit(1)
            # 根据网络结构信息，初始化指定结构的网络
            if checkpoint.get('network_structure', NetStruct.NORMAL) == NetStruct.DUELING:
                variable_qnetwork = FlappyDuelingQNet().to(self.device)
            else:
                variable_qnetwork = FlappyQNet().to(self.device)
            # 加载网络参数
            variable_qnetwork.load_state_dict(checkpoint.get('state_dict', None))
        else:
            if 'Dueling DQN' in self.training_setting.advanced_method:
                variable_qnetwork = FlappyDuelingQNet().to(self.device)
            else:
                variable_qnetwork = FlappyQNet().to(self.device)

        # 初始化target q-network
        target_qnetwork = copy.deepcopy(variable_qnetwork).to(self.device)

        # 初始化优化器和损失函数
        optimizer = torch.optim.RMSprop(variable_qnetwork.parameters(), lr=self.training_setting.lr)
        ceriterion = torch.nn.MSELoss()

        # 初始化游戏
        game_render_setting = flappybird.settings.RenderSetting()
        game_render_setting.set_mode('hidden')
        flappyBird_game_manager = FlappyBirdGameManager(game_render_setting)
        flappyBird_game_manager.set_player_computer()

        """
        训练开始前，随机选取action操作小鸟，并将数据保存起来
        随机操作的次数取决于training_setting.observation
        """
        action = [1, 0]
        o, r, terminal = flappyBird_game_manager.frame_step(action)
        o = self.frame_preprocess(o)
        agent.reset_state()
        for i in range(self.training_setting.observation):
            current_state = agent.current_state
            action = agent.get_action_based_on_fixed_pr()
            o, r, terminal = flappyBird_game_manager.frame_step(action)
            o = self.frame_preprocess(o)
            next_state = agent.update_current_state(o)
            self.replay_memory.push(current_state, action, r, next_state, terminal)

        # start training
        flappyBird_game_manager.game_reset()
        epsilon = self.training_setting.epsilon_greedy.init_e

        flag_update_target_qnetwork = 0

        # 注意episode从0开始编号，所以训练次数可以在max_episode的基础上+1，否则最后的一部分训练结果没有机会保存下来
        for episode in range(self.training_setting.max_episode + 1):
            agent.time_step = 0
            agent.reset_state()
            total_reward = 0.
            # ------beginning of an episode------
            while True:
                # 模型依据自身经验决定这一帧采取的action，传入gamestate，获得这一帧的观测图像、奖励值、游戏是否中止
                # 决定action的方法受到exploration方式的影响，目前支持Epsilon Greedy和Boltzmann Exploration
                action = agent.get_action_based_on_exploration(
                    variable_qnetwork,
                    exploration_method=ExplorationMethod.BOLTZMANN_EXPLORATION if self.training_setting.exploration_method == 'Boltzmann Exploration' else ExplorationMethod.EPSILON_GREEDY,
                    epsilon=epsilon)
                o_next, r, terminal = flappyBird_game_manager.frame_step(action)
                total_reward += self.training_setting.gamma**agent.time_step * r

                # 对这一帧图像做预处理
                o_next = self.frame_preprocess(o_next)

                current_state = agent.current_state
                next_state = agent.update_current_state(o_next)

                # 保存数据，模型的time_step计数器+1
                self.replay_memory.push(current_state, action, r, next_state, terminal)
                agent.increase_time_step()

                # Step 1: obtain random minibatch from replay memory
                minibatch = self.replay_memory.sample(self.training_setting.batch_size)
                state_batch = np.array([data[0] for data in minibatch])
                action_batch = np.array([data[1] for data in minibatch])
                reward_batch = np.array([data[2] for data in minibatch])
                next_state_batch = np.array([data[3] for data in minibatch])

                state_batch_var = Variable(torch.from_numpy(state_batch)).to(self.device)
                with torch.no_grad():
                    next_state_batch_var = Variable(torch.from_numpy(next_state_batch)).to(self.device)

                """
                Step 2: calculate y
                """

                r"""
                计算variable_qnetwork的Q值$Q(s_i, a_i)$与目标y值，更新网络权重使Q接近y（回归问题）
                原始方法：
                $y = r_t + \underset{a}{max}\hat{Q}(s_{t+1}, a)$
                进阶方法(Double DQN)：
                $y = r_t + Q'(s_{t+1}, arg\underset{a}{max}Q(s_{t+1}, a))$
                """
                q_of_next_state = variable_qnetwork(next_state_batch_var)
                y = reward_batch.astype(np.float32)

                if 'Double DQN' in self.training_setting.advanced_method:
                    # max_q.shape: Tensor([32])
                    # max_q_index.shape: Tensor([32]), max_q_index每个位置的值只会是0或1
                    # target_qnetwork.forward(next_state_batch_var).shape: Tensor([32, 2]), 二维数组
                    # 用二维索引的方式，把target_qnetwork算出的Q表里，每行指定索引位置的值取出来
                    # recalculated_q.shape: Tensor([32])
                    _, max_q_index = torch.max(q_of_next_state, dim=1)
                    # TODO: 下面这个索引方式有无更好的方式代替
                    recalculated_q = target_qnetwork(next_state_batch_var)[torch.arange(
                        0, self.training_setting.batch_size), max_q_index]
                    for i in range(self.training_setting.batch_size):
                        if not minibatch[i][4]:
                            y[i] += self.training_setting.gamma * recalculated_q.data[i].item()
                else:
                    max_q, _ = torch.max(q_of_next_state, dim=1)

                    for i in range(self.training_setting.batch_size):
                        if not minibatch[i][4]:
                            y[i] += self.training_setting.gamma * max_q.data[i].item()

                action_batch_var = Variable(torch.from_numpy(action_batch)).to(self.device)
                q_of_current_state = variable_qnetwork(state_batch_var)
                q_of_current_state = torch.sum(torch.mul(action_batch_var, q_of_current_state), dim=1)
                y = Variable(torch.from_numpy(y)).to(self.device)

                # 更新网络参数
                loss = ceriterion(q_of_current_state, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                """
                每经过一定次数的更新，将target_qnetwork更新为当前的variable_qnetwork
                """
                if (flag_update_target_qnetwork := ((flag_update_target_qnetwork + 1) %
                                                    self.training_setting.update_target_qnetwork_freq)) == 0:
                    target_qnetwork = copy.deepcopy(variable_qnetwork)

                # when the bird dies, the episode ends
                if terminal:
                    break

            # ------end of an episode------

            self.generate_log(message='episode: {}, epsilon: {:.4f}, max time step: {}, total reward: {:.6f}'.format(
                episode, epsilon, agent.time_step, total_reward),
                level='info', location=os.path.split(__file__)[1])

            # 经过一次episode后，降低epsilon的值
            if epsilon > self.training_setting.epsilon_greedy.final_e:
                delta = (self.training_setting.epsilon_greedy.init_e - self.training_setting.epsilon_greedy.final_e) / \
                    self.training_setting.exploration
                epsilon -= delta

            """
            每经过一定次数的episode，测试训练后模型的效果(具体次数为training_setting.test_model_freq，默认值见程序入口)
            如果训练后的模型效果经过估计优于训练前的模型，将其保存起来，并且接下来的训练过程基于这个新的模型进行
            否则，按照training_setting.save_checkpoint_freq的值，每隔一定数量的episode保存一次模型，不管这个模型是否是当前最优的
            """
            # 用于查看当前episode是否保存了检查点的标志变量
            checkpoint_saved = False
            # 用于检查当前episode的模型是否被评估的标志变量
            model_evaluated = False

            # case1: 测试模型
            if episode % self.training_setting.test_model_freq == 0:
                if not model_evaluated:
                    avg_time_step = self.evaluate_avg_time_step(agent, variable_qnetwork)
                    self.generate_log(message='testing: episode: {}, average time step: {}'.format(
                        episode, avg_time_step),
                        level='info', location=os.path.split(__file__)[1])
                    model_evaluated = True
                if avg_time_step > best_time_step:
                    best_time_step = avg_time_step
                    model_dict = {
                        'episode': episode,
                        'epsilon': epsilon,
                        'state_dict': variable_qnetwork.state_dict(),
                        'network_structure': NetStruct.DUELING if isinstance(variable_qnetwork, FlappyDuelingQNet) else NetStruct.NORMAL,
                        'time_step': best_time_step,
                    }
                    self.file_handler.save(model_dict, name='checkpoint-episode-%d.pth.tar' % episode)
                    checkpoint_saved = True
                    self.generate_log(message='save the best checkpoint by far, episode={}, average time step={:.2f}'.format(
                        episode, avg_time_step),
                        level='info', location=os.path.split(__file__)[1])
                    # 把当前最佳的模型信息另外在根目录保存一份
                    self.file_handler.save(model_dict, 'model_best.pth.tar', './')

            # case2: 保存检查点
            if episode % self.training_setting.save_checkpoint_freq == 0 and not checkpoint_saved:
                if not model_evaluated:
                    avg_time_step = self.evaluate_avg_time_step(agent, variable_qnetwork)
                    self.generate_log(message='testing: episode: {}, average time step: {}'.format(
                        episode, avg_time_step),
                        level='info', location=os.path.split(__file__)[1])
                    model_evaluated = True
                model_dict = {
                    'episode': episode,
                    'epsilon': epsilon,
                    'state_dict': variable_qnetwork.state_dict(),
                    'network_structure': NetStruct.DUELING if isinstance(variable_qnetwork, FlappyDuelingQNet) else NetStruct.NORMAL,
                    'time_step': avg_time_step,
                }
                self.file_handler.save(model_dict, name='checkpoint-episode-%d.pth.tar' % episode)
                self.generate_log(message='save a checkpoint at a preset frequency, episode={}, average time step={:.2f}'.format(
                    episode, avg_time_step),
                    level='info', location=os.path.split(__file__)[1])
                checkpoint_saved = True

            # case3: 不评估，继续下一个episode
            else:
                continue
        pass

    def evaluate_avg_time_step(self, agent: FlappyAgent, network: torch.nn.Module, test_episode_num=45):
        '''
        评估当前模型在数次游戏中坚持的平均时间，用于测试当前模型的游戏效果

        具体步骤：

        设n为测试游戏次数
        模型在与训练相同的gamestate下游玩n次，游玩过程采取的action全部为依据自身数据选择的，而非随机选取。n次游戏结束后，返回平均游戏时间。

        原作者设定n=5
        目前设定n=45，舍弃最好与最差的10次，试图降低某些极端情况的影响

        :param agent: 智能体
        :param episode: current training episode
        :returns avg_time_step: 模型在n次游戏中坚持的平均时间
        '''
        time_step_list = []
        game_render_setting = flappybird.settings.RenderSetting()
        game_render_setting.set_mode('hidden')
        flappyBird_game_manager = FlappyBirdGameManager(game_render_setting)
        flappyBird_game_manager.set_player_computer()

        for test_case in range(test_episode_num):
            agent.time_step = 0
            agent.reset_state()
            observation_frame, reward, terminal = flappyBird_game_manager.frame_step([1, 0])
            observation_frame = self.frame_preprocess(observation_frame)
            while True:
                action = agent.get_optim_action(network)
                observation_frame, reward, terminal = flappyBird_game_manager.frame_step(action)
                if terminal:
                    break
                observation_frame = self.frame_preprocess(observation_frame)
                agent.update_current_state(observation_frame)
                agent.increase_time_step()
            time_step_list.append(agent.time_step)
        time_step_list = sorted(time_step_list)[10:-10]
        # time_step_list = sorted(time_step_list)
        avg_time_step = sum(time_step_list) / len(time_step_list)

        return avg_time_step

    def play_game(self, player, args=None):
        '''
        运行游戏，并根据传入参数确定游戏模式
        '''
        # try:
        if player == 'human':
            game_render_setting = flappybird.settings.RenderSetting()
            game_render_setting.set_mode('human')
            game = FlappyBirdGameManager(game_render_setting)
            game.set_player_human()
            game.start_game_by_human()
        elif player == 'computer':
            self.play_game_with_model(model_file_path=args.model_path, cuda=args.cuda)

    def play_game_with_model(self, model_file_path, cuda=False):
        '''
        使用一个训练过的模型玩flappybird游戏

        :param weight: model file name containing weight of dqn
        :param best: if the model is best or not
        '''

        device = torch.device('cuda:0') if cuda else torch.device('cpu')

        """
        从磁盘中加载模型信息
        根据信息初始化agent与network
        并把加载的网络参数赋给network
        """
        self.generate_log(
            message='Load pretrained model file: ' + model_file_path,
            level='info', location=os.path.split(__file__)[1])
        try:
            checkpoint = self.file_handler.load(model_file_path)
        except BaseException as e:
            self.generate_log(message='Error raised when loading model. Type: {}, Description: {}'.format(type(e), e),
                              level='error', location=os.path.split(__file__)[1])
            sys.exit(1)

        # 根据网络结构信息，初始化指定结构的网络
        if checkpoint.get('network_structure', NetStruct.NORMAL) == NetStruct.DUELING:
            qnetwork = FlappyDuelingQNet().to(device)
        else:
            qnetwork = FlappyQNet().to(device)
        # 加载网络参数
        qnetwork.load_state_dict(checkpoint.get('state_dict', None))

        agent = FlappyAgent(0, device)
        agent.reset_state()

        # 初始化游戏
        game_render_setting = flappybird.settings.RenderSetting()
        game_render_setting.set_mode('human')
        flappyBird_game_manager = FlappyBirdGameManager(game_render_setting)
        flappyBird_game_manager.set_player_computer()

        while True:
            # agent选择动作
            action = agent.get_optim_action(qnetwork)
            # 与游戏环境交互，拿到观测到的一帧图像、奖励，得知游戏是否停止
            observation_frame, reward, terminal = flappyBird_game_manager.frame_step(action)
            if terminal:
                break
            # 更新agent当前观测的状态
            observation_frame = self.frame_preprocess(observation_frame)
            agent.update_current_state(observation_frame)

            agent.increase_time_step()

        self.generate_log(
            message='total time step is {}'.format(agent.time_step),
            level='info', location=os.path.split(__file__)[1])
