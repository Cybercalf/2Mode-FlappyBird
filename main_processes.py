import sys
import os
import time
import random
import shutil
import copy

import numpy as np
import PIL.Image
from torch.autograd import Variable
import torch
import torch.nn
import torch.optim

from network.qnetwork.flappybird_qnetwork import FlappyBirdQNetwork as QNetwork
from util.logger.logger_subject import LoggerSubject
from util.logger.logger_observer import ConsoleLoggerOberver, FileLoggerObserver
import flappybird.settings
from flappybird.game_manager import GameManager as FlappyBirdGameManager


class ProgramManager(LoggerSubject):
    '''
    程序管理器，负责运行游戏、训练模型等一系列任务
    '''

    def __init__(self):
        super().__init__()
        """
        在初始化时，自动注册日志打印器，之后输出日志时，直接调用父类的打印方法即可
        """
        self.console_info_logger = ConsoleLoggerOberver()
        self.console_error_logger = ConsoleLoggerOberver()
        self.register_observer(self.console_info_logger, 'info')
        self.register_observer(self.console_error_logger, 'error')

        self.file_info_logger = FileLoggerObserver('{}.log'.format(time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())))
        self.file_error_logger = FileLoggerObserver('{}.log'.format(time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())))
        self.register_observer(self.file_info_logger, 'info')
        self.register_observer(self.file_error_logger, 'error')

    def load_training_setting(self, training_setting):
        '''
        加载训练设置
        '''
        self.training_setting = training_setting
        if self.training_setting.cuda is True:
            self.device = torch.device('cuda:0')
        else:
            self.device = torch.device('cpu')

    def preprocess(self, frame, image_size_after_resize=(72, 128)):
        '''
        对输入的帧图像做预处理

        输入图像：512*288，rgb彩色图像

        预处理步骤：

        1.降采样，使图像尺寸变为128*72

        2.将图像转换为灰度图像

        3.将图像每个像素的灰度值映射为0或1
        '''
        # TODO: 解释为什么这里的降采样尺寸是(72, 128)，而创建空Tensor时采用(128, 72)
        im = PIL.Image.fromarray(frame).resize(
            image_size_after_resize).convert(mode='L')
        out = np.asarray(im).astype(np.float32)
        out[out <= 1.] = 0.0
        out[out > 1.] = 1.0
        return out

    def save_checkpoint(self, state, is_best, filepath='checkpoint.pth.tar'):
        '''
        经过一定训练次数后，保存当前模型，并检查其是否是目前最优的模型。

        如果是，系统会再拷贝一份模型出来，用一个醒目的命名告诉用户这是目前效果最好的模型

        :param state: checkpoint state: model weight and other info binding by user

        :param is_best: if the checkpoint is the best currently. If it is, then save as a best model
        '''
        torch.save(state, filepath)
        if is_best:
            shutil.copyfile(filepath, 'model_best.pth.tar')

    def load_checkpoint(self, filepath, model):
        '''
        从磁盘中加载之前保存的检查点模型

        :param filename: model file name

        :param model: DQN model
        '''

        try:
            checkpoint = torch.load(filepath)
        except FileNotFoundError as file_not_found_error:
            # 若从磁盘中找不到要加载的文件，生成错误日志并退出
            self.generate_log(message='Error raised when loading model. Type: {}, Description: {}'.format(
                type(file_not_found_error), file_not_found_error),
                level='error', location=os.path.split(__file__)[1])
            sys.exit(1)
        # 如果磁盘文件以gpu的方式存储，直接加载到cpu上可能会出现异常，检测到异常时尝试使用另一种方法加载
        except BaseException as e:
            self.generate_log(
                message='Exception raised when loading model. Type: {}, Description: {}. Trying another way to load model.'.format(type(e), e), level='info', location=os.path.split(__file__)[1])
            # load weight saved on gpu device to cpu device
            # see
            # https://discuss.pytorch.org/t/on-a-cpu-device-how-to-load-checkpoint-saved-on-gpu-device/349/3
            checkpoint = torch.load(
                filepath, map_location=lambda storage, loc: storage)

        """
        调试输出(debug)
        输出加载的模型在之前经过了多少训练，用于探索的epsilon值为多少
        """
        episode = checkpoint['episode']
        epsilon = checkpoint['epsilon']
        # is_best = checkpoint['is_best_model_by_far']
        # 调试输出
        self.generate_log(message='pretrained episode = {}'.format(episode),
                          level='info', location=os.path.split(__file__)[1])
        self.generate_log(message='pretrained epsilon = {}'.format(epsilon),
                          level='info', location=os.path.split(__file__)[1])
        # self.generate_log(message='model.is_best: {}'.format(is_best),
        #                   level='info', location=os.path.split(__file__)[1])

        model.load_state_dict(checkpoint['state_dict'])

        """
        调试输出(debug)
        输出加载的模型在训练过程中操纵小鸟飞行的最长时间
        """

        # ------ begin of TODO ------
        # best_time_step是过去checkpoint使用的参数。在重新训练模型之后，请将此部分语句改为：
        # time_step = checkpoint.get('time_step', None)

        time_step = checkpoint.get('best_time_step', None)
        if time_step is None:
            time_step = checkpoint.get('time_step', None)

        # ------ end of TODO ------

        if time_step is None:
            self.generate_log(message='Error: the model to be loaded has no attribute named "time_step".',
                              level='error', location=os.path.split(__file__)[1])
            sys.exit(1)
        self.generate_log(message='pretrained time step = {}'.format(time_step),
                          level='info', location=os.path.split(__file__)[1])

        return episode, epsilon, time_step

    def train_model(self, training_setting):
        '''
        加载训练设置并训练模型
        '''
        self.load_training_setting(training_setting)
        self.training_process()

    def training_process(self):
        '''
        训练模型的核心过程

        :param options: 通用设置，如模型路径、cuda支持等
        :param training_setting: 训练过程中使用的各项参数

        :param model: DQN model
        :param resume: resume previous model

        :param lr: learning rate
        :param max_episode: maximum episode
        :param model_name: checkpoint file name
        '''

        """
        创建文件夹，用于存放训练过程中保存的模型
        """
        checkpoint_folder_name = './model/checkpoint_' + \
            time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()) + '/'
        os.makedirs(checkpoint_folder_name, exist_ok=True)

        # 记录模型的最好游玩效果，用time_step量化
        best_time_step = 0.

        """
        初始化QNetwork
        1.一个在训练过程中经常变化权重的QNetwork，用于训练
        """
        variable_qnetwork = QNetwork(epsilon=self.training_setting.epsilon_greedy.init_e,
                                     mem_size=self.training_setting.memory_size, cuda=self.training_setting.cuda)

        """
        检查训练过程是否基于一个给定的模型开始
        """
        if self.training_setting.resume:
            # 2024.01.04
            # 目前程序未传入--model参数时，当做人类游玩游戏处理，所以下面这个异常理论上不会被触发
            if self.training_setting.model_path is None:
                self.generate_log(message='Error: when resume, you should give weight file name.',
                                  level='error', location=os.path.split(__file__)[1])
                return
            self.generate_log(message='load previous model weight: {}'.format(self.training_setting.model_path),
                              level='info', location=os.path.split(__file__)[1])
            _, _, best_time_step = self.load_checkpoint(
                self.training_setting.model_path, variable_qnetwork)

        """
        初始化各参数
        用于传入action返回状态数据的gamestate
        优化器optimizer
        误差函数ceriterion
        观测数据（这一帧的游戏画面）o
        这一帧的奖励r
        游戏在这一帧是否结束terminal
        """
        gamestate_setting = flappybird.settings.Setting()
        gamestate_setting.set_mode(mode='train')
        flappyBird_game_manager = FlappyBirdGameManager(gamestate_setting)
        flappyBird_game_manager.set_player_computer()

        optimizer = torch.optim.RMSprop(variable_qnetwork.parameters(), lr=self.training_setting.lr)
        ceriterion = torch.nn.MSELoss()

        action = [1, 0]
        o, r, terminal = flappyBird_game_manager.frame_step(action)
        o = self.preprocess(o)
        variable_qnetwork.set_initial_state()

        # TODO: 加入device变量，让转移到cuda设备的操作用.to(device)而不是.cuda()实现
        variable_qnetwork = variable_qnetwork.to(self.device)

        """
        训练开始前，随机选取action操作小鸟，并将数据保存起来
        随机操作的次数取决于training_setting.observation
        """
        for i in range(self.training_setting.observation):
            action = variable_qnetwork.get_action_randomly()
            o, r, terminal = flappyBird_game_manager.frame_step(action)
            o = self.preprocess(o)
            variable_qnetwork.store_transition(o, action, r, terminal)

        # start training
        """
        初始化QNetwork
        2.一个仅在某些规定时刻更改的QNetwork，用于计算前一个QNetwork产生Q值的目标
        """
        target_qnetwork = copy.deepcopy(variable_qnetwork)

        # 注意episode从0开始编号，所以训练次数可以在max_episode的基础上+1，否则最后的一部分训练结果没有机会保存下来
        for episode in range(self.training_setting.max_episode + 1):
            variable_qnetwork.time_step = 0
            variable_qnetwork.set_train()
            total_reward = 0.

            # ------beginning of an episode------
            while True:
                optimizer.zero_grad()

                # 模型依据自身经验决定这一帧采取的action，传入gamestate，获得这一帧的观测图像、奖励值、游戏是否中止
                # 决定action的方法受到exploration方式的影响，目前支持Epsilon Greedy和Boltzmann Exploration
                action = variable_qnetwork.get_action(setting=self.training_setting)
                o_next, r, terminal = flappyBird_game_manager.frame_step(action)
                total_reward += self.training_setting.gamma**variable_qnetwork.time_step * r

                # 对这一帧图像做预处理
                o_next = self.preprocess(o_next)

                # 保存数据，模型的time_step计数器+1
                variable_qnetwork.store_transition(o_next, action, r, terminal)
                variable_qnetwork.increase_time_step()

                # Step 1: obtain random minibatch from replay memory
                minibatch = random.sample(
                    variable_qnetwork.replay_memory, self.training_setting.batch_size)
                state_batch = np.array([data[0] for data in minibatch])
                action_batch = np.array([data[1] for data in minibatch])
                reward_batch = np.array([data[2] for data in minibatch])
                next_state_batch = np.array([data[3] for data in minibatch])
                state_batch_var = Variable(torch.from_numpy(state_batch))
                with torch.no_grad():
                    next_state_batch_var = Variable(
                        torch.from_numpy(next_state_batch))
                state_batch_var = state_batch_var.to(self.device)
                next_state_batch_var = next_state_batch_var.to(self.device)

                """
                Step 2: calculate y
                TODO: 看懂y值的计算步骤
                """
                q_value_next = variable_qnetwork.forward(next_state_batch_var)

                q_value = variable_qnetwork.forward(state_batch_var)

                y = reward_batch.astype(np.float32)

                r"""
                计算variable_qnetwork的Q值$Q(s_i, a_i)$与目标y值，更新网络权重使Q接近y（回归问题）
                原始方法：
                $y = r_t + \underset{a}{max}\hat{Q}(s_{t+1}, a)$
                进阶方法(Double DQN)：
                $y = r_t + Q'(s_{t+1}, arg\underset{a}{max}Q(s_{t+1}, a))$
                """
                if 'Double DQN' in self.training_setting.advanced_method:
                    # max_q.shape: Tensor([32])
                    # max_q_index.shape: Tensor([32]), max_q_index每个位置的值只会是0或1
                    # target_qnetwork.forward(next_state_batch_var).shape: Tensor([32, 2]), 二维数组
                    # 用二维索引的方式，把target_qnetwork算出的Q表里，每行指定索引位置的值取出来
                    # recalculated_q.shape: Tensor([32])
                    _, max_q_index = torch.max(q_value_next, dim=1)
                    # TODO: 下面这个索引方式有无更好的方式代替
                    recalculated_q = target_qnetwork.forward(next_state_batch_var)[
                        torch.arange(0, self.training_setting.batch_size), max_q_index]
                    for i in range(self.training_setting.batch_size):
                        if not minibatch[i][4]:
                            y[i] += self.training_setting.gamma * recalculated_q.data[i].item()
                else:
                    max_q, _ = torch.max(q_value_next, dim=1)

                    for i in range(self.training_setting.batch_size):
                        if not minibatch[i][4]:
                            y[i] += self.training_setting.gamma * max_q.data[i].item()

                y = Variable(torch.from_numpy(y))

                action_batch_var = Variable(torch.from_numpy(action_batch))

                y = y.to(self.device)
                action_batch_var = action_batch_var.to(self.device)

                q_value = torch.sum(
                    torch.mul(
                        action_batch_var,
                        q_value),
                    dim=1)

                loss = ceriterion(q_value, y)
                loss.backward()

                optimizer.step()
                # when the bird dies, the episode ends
                if terminal:
                    break

            # ------end of an episode------

            self.generate_log(message='episode: {}, epsilon: {:.4f}, max time step: {}, total reward: {:.6f}'.format(
                episode, variable_qnetwork.epsilon, variable_qnetwork.time_step, total_reward),
                level='info', location=os.path.split(__file__)[1])

            # 经过一次episode后，降低epsilon的值
            if variable_qnetwork.epsilon > self.training_setting.epsilon_greedy.final_e:
                delta = (self.training_setting.epsilon_greedy.init_e - self.training_setting.epsilon_greedy.final_e) / \
                    self.training_setting.exploration
                variable_qnetwork.epsilon -= delta

            """
            每经过一定次数的episode，将target_qnetwork更新为当前的variable_qnetwork
            TODO: 找到一个更好的更新target_qnetwork的时机
            """
            if episode % self.training_setting.update_target_qnetwork_freq == 0:
                target_qnetwork = copy.deepcopy(variable_qnetwork)

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
                    avg_time_step = self.evaluate_avg_time_step(variable_qnetwork, episode)
                    model_evaluated = True
                if avg_time_step > best_time_step:
                    best_time_step = avg_time_step
                    self.save_checkpoint({
                        'episode': episode,
                        'epsilon': variable_qnetwork.epsilon,
                        'state_dict': variable_qnetwork.state_dict(),
                        'is_best_model_by_far': True,
                        'time_step': best_time_step,
                    }, is_best=True, filepath=checkpoint_folder_name + 'checkpoint-episode-%d.pth.tar' % episode)
                    checkpoint_saved = True
                    self.generate_log(message='save the best checkpoint by far, episode={}, average time step={:.2f}'.format(
                        episode, avg_time_step),
                        level='info', location=os.path.split(__file__)[1])

            # case2: 保存检查点
            if episode % self.training_setting.save_checkpoint_freq == 0 and not checkpoint_saved:
                if not model_evaluated:
                    avg_time_step = self.evaluate_avg_time_step(variable_qnetwork, episode)
                    model_evaluated = True
                self.save_checkpoint({
                    'episode': episode,
                    'epsilon': variable_qnetwork.epsilon,
                    'state_dict': variable_qnetwork.state_dict(),
                    'is_best_model_by_far': False,
                    'time_step': avg_time_step,
                }, is_best=False, filepath=checkpoint_folder_name + 'checkpoint-episode-%d.pth.tar' % episode)
                self.generate_log(message='save a normal checkpoint, episode={}, average time step={:.2f}'.format(
                    episode, avg_time_step),
                    level='info', location=os.path.split(__file__)[1])
                checkpoint_saved = True

            # case3: 不评估，继续下一个episode
            else:
                continue

    def evaluate_avg_time_step(
            self, model, current_episode, test_episode_num=20):
        '''
        评估当前模型在数次游戏中坚持的平均时间，用于测试当前模型的游戏效果

        具体步骤：

        设n为测试游戏次数
        模型在与训练相同的gamestate下游玩n次，游玩过程采取的action全部为依据自身数据选择的，而非随机选取。n次游戏结束后，返回平均游戏时间。

        原作者设定n=5
        TODO: debug. 设定n=20，舍弃最好与最差的5次

        :param model: dqn model
        :param episode: current training episode
        :returns avg_time_step: 模型在n次游戏中坚持的平均时间
        '''
        model.set_eval()
        # avg_time_step = 0.
        time_step_list = []
        gamestate_setting = flappybird.settings.Setting()
        gamestate_setting.set_mode(mode='train')
        flappyBird_game_manager = FlappyBirdGameManager(gamestate_setting)
        flappyBird_game_manager.set_player_computer()
        for test_case in range(test_episode_num):
            model.time_step = 0
            o, r, terminal = flappyBird_game_manager.frame_step([1, 0])
            o = self.preprocess(o)
            model.set_initial_state()
            while True:
                action = model.get_optim_action()
                o, r, terminal = flappyBird_game_manager.frame_step(action)
                if terminal:
                    break
                o = self.preprocess(o)
                model.current_state = np.append(
                    model.current_state[1:, :, :], o.reshape((1,) + o.shape), axis=0)
                model.increase_time_step()
            time_step_list.append(model.time_step)
        time_step_list = sorted(time_step_list)[5:-5]
        avg_time_step = sum(time_step_list) / len(time_step_list)

        self.generate_log(message='testing: episode: {}, average time step: {}'.format(
            current_episode, avg_time_step),
            level='info', location=os.path.split(__file__)[1])

        return avg_time_step

    def play_game(self, player, args=None):
        '''
        运行游戏，并根据传入参数确定游戏模式
        '''
        try:
            if player == 'human':
                gamestate_setting = flappybird.settings.Setting()
                gamestate_setting.set_mode('play')
                game = FlappyBirdGameManager(setting=gamestate_setting)
                game.set_player_human()
                game.start_game_by_human()
            elif player == 'computer':
                self.play_game_with_model(
                    model_file_path=args.model_path,
                    cuda=args.cuda)
        except AttributeError as e:
            self.generate_log(message='Error caught. Type: {}, description: {}'.format(
                type(e), e),
                level='error', location=os.path.split(__file__)[1])

    def play_game_with_model(self, model_file_path, cuda=False):
        '''
        使用一个训练过的模型玩flappybird游戏

        :param weight: model file name containing weight of dqn
        :param best: if the model is best or not
        '''

        # 加载模型
        self.generate_log(
            message='load pretrained model file: ' + model_file_path,
            level='info', location=os.path.split(__file__)[1])
        model = QNetwork(epsilon=0., mem_size=0, cuda=cuda)
        self.load_checkpoint(model_file_path, model)
        model.set_initial_state()
        if cuda:
            model = model.cuda()
        model.set_eval()

        # 初始化游戏
        gamestate_setting = flappybird.settings.Setting()
        gamestate_setting.set_mode('play')
        flappyBird_game_manager = FlappyBirdGameManager(gamestate_setting)
        flappyBird_game_manager.set_player_computer()

        while True:
            action = model.get_optim_action()
            o, r, terminal = flappyBird_game_manager.frame_step(action)
            if terminal:
                break
            o = self.preprocess(o)

            model.current_state = np.append(
                model.current_state[1:, :, :], o.reshape((1,) + o.shape), axis=0)

            model.increase_time_step()

        self.generate_log(
            message='total time step is {}'.format(model.time_step),
            level='info', location=os.path.split(__file__)[1])
