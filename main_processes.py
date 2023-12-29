# TODO: 尝试把只在一个func内用到的软编码值独立出来，组成一个或多个setting类，也许能减少变量创建的开销
import PIL.Image
from torch.autograd import Variable
import torch.optim
import torch.nn
import torch
import random
import numpy as np
import shutil
import network
import game.dqn_training_gamestate as gamestate_for_training
import game.dqn_mode_gamestate as gamestate_for_playing
import sys
import os
import time
sys.path.append("game/")


def preprocess(frame, image_size_after_resize=(72, 128)):
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


def save_checkpoint(state, is_best, filepath='checkpoint.pth.tar'):
    '''
    经过一定训练次数后，保存当前模型，并检查其是否是目前最优的模型。

    如果是，系统会再拷贝一份模型出来，用一个醒目的命名告诉用户这是目前效果最好的模型

    :param state: checkpoint state: model weight and other info binding by user

    :param is_best: if the checkpoint is the best currently. If it is, then save as a best model
    '''
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, 'model_best.pth.tar')


def load_checkpoint(filepath, model):
    '''
    从磁盘中加载之前保存的检查点模型

    :param filename: model file name

    :param model: DQN model
    '''

    try:
        checkpoint = torch.load(filepath)
    # 如果磁盘文件以gpu的方式存储，直接加载到cpu上会出现异常，检测到异常时使用另一种方法加载
    except BaseException:
        # load weight saved on gpy device to cpu device
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
    print('[main_processes.py] pretrained episode = {}'.format(episode))
    print('[main_processes.py] pretrained epsilon = {}'.format(epsilon))

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
        print(
            '[main_processes.py] Error: the model to be loaded has no attribute named "time_step".')
        sys.exit(1)
    print('[main_processes.py] pretrained time step = {}'.format(time_step))

    return episode, epsilon, time_step


def train_model(model, options, resume):
    '''
    训练模型的核心过程

    :param model: DQN model
    :param options: 训练使用的各项参数
    :param resume: resume previous model

    :param lr: learning rate
    :param max_episode: maximum episode
    :param model_name: checkpoint file name
    '''

    """
    创建文件夹，用于存放训练过程中保存的模型
    """
    checkpoint_folder_name = './model/checkpoint_' + time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()) + '/'
    os.makedirs(checkpoint_folder_name, exist_ok=True)

    best_time_step = 0.
    if resume:
        if options.weight is None:
            print(
                '[main_processes.py] Error: when resume, you should give weight file name.')
            return
        print(
            '[main_processes.py] load previous model weight: {}'.format(
                options.weight))
        _, _, best_time_step = load_checkpoint(options.weight, model)

    """
    初始化各参数
    用于传入action返回状态数据的gamestate
    优化器optimizer
    误差函数ceriterion
    观测数据（这一帧的游戏画面）o
    这一帧的奖励r
    游戏在这一帧是否结束terminal
    """
    flappyBird = gamestate_for_training.GameState()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=options.lr)
    ceriterion = torch.nn.MSELoss()

    action = [1, 0]
    o, r, terminal = flappyBird.frame_step(action)
    o = preprocess(o)
    model.set_initial_state()

    if options.cuda:
        model = model.cuda()

    """
    训练开始前，随机选取action操作小鸟，并将数据保存起来
    随机操作的次数取决于options.observation
    """
    for i in range(options.observation):
        action = model.get_action_randomly()
        o, r, terminal = flappyBird.frame_step(action)
        o = preprocess(o)
        model.store_transition(o, action, r, terminal)

    # start training
    # 注意episode从0开始编号，所以训练次数可以在max_episode的基础上+1，否则最后的一部分训练结果没有机会保存下来
    for episode in range(options.max_episode + 1):
        model.time_step = 0
        model.set_train()
        total_reward = 0.

        # ------beginning of an episode------
        while True:
            optimizer.zero_grad()

            # 模型依据自身经验决定这一帧采取的action，传入gamestate，获得这一帧的观测图像、奖励值、游戏是否中止
            action = model.get_action()
            o_next, r, terminal = flappyBird.frame_step(action)
            total_reward += options.gamma**model.time_step * r

            # 对这一帧图像做预处理
            o_next = preprocess(o_next)

            # 保存数据，模型的time_step计数器+1
            model.store_transition(o_next, action, r, terminal)
            model.increase_time_step()

            # Step 1: obtain random minibatch from replay memory
            minibatch = random.sample(model.replay_memory, options.batch_size)
            state_batch = np.array([data[0] for data in minibatch])
            action_batch = np.array([data[1] for data in minibatch])
            reward_batch = np.array([data[2] for data in minibatch])
            next_state_batch = np.array([data[3] for data in minibatch])
            state_batch_var = Variable(torch.from_numpy(state_batch))
            with torch.no_grad():
                next_state_batch_var = Variable(
                    torch.from_numpy(next_state_batch))
            if options.cuda:
                state_batch_var = state_batch_var.cuda()
                next_state_batch_var = next_state_batch_var.cuda()

            """
            Step 2: calculate y
            TODO: 看懂y值的计算步骤
            """
            q_value_next = model.forward(next_state_batch_var)

            q_value = model.forward(state_batch_var)

            y = reward_batch.astype(np.float32)
            max_q, _ = torch.max(q_value_next, dim=1)

            for i in range(options.batch_size):
                if not minibatch[i][4]:
                    y[i] += options.gamma * max_q.data[i].item()

            y = Variable(torch.from_numpy(y))
            action_batch_var = Variable(torch.from_numpy(action_batch))
            if options.cuda:
                y = y.cuda()
                action_batch_var = action_batch_var.cuda()
            q_value = torch.sum(torch.mul(action_batch_var, q_value), dim=1)

            loss = ceriterion(q_value, y)
            loss.backward()

            optimizer.step()
            # when the bird dies, the episode ends
            if terminal:
                break

        # ------end of an episode------

        print('[main_processes.py] episode: {}, epsilon: {:.4f}, max time step: {}, total reward: {:.6f}'.format(
            episode, model.epsilon, model.time_step, total_reward))

        # 经过一次episode后，降低epsilon的值
        if model.epsilon > options.final_e:
            delta = (options.init_e - options.final_e) / options.exploration
            model.epsilon -= delta

        """
        每经过一定次数的episode，测试训练后模型的效果(具体次数为options.test_model_freq，默认值见程序入口)
        如果训练后的模型效果经过估计优于训练前的模型，将其保存起来，并且接下来的训练过程基于这个新的模型进行
        否则，按照options.save_checkpoint_freq的值，每隔一定数量的episode保存一次模型，不管这个模型是否是当前最优的
        """
        # 用于查看当前episode是否保存了检查点的标志变量
        checkpoint_saved = False
        # 用于检查当前episode的模型是否被评估的标志变量
        model_evaluated = False
        # case1: 测试模型
        if episode % options.test_model_freq == 0:
            if not model_evaluated:
                avg_time_step = evaluate_avg_time_step(model, episode)
                model_evaluated = True
            if avg_time_step > best_time_step:
                best_time_step = avg_time_step
                save_checkpoint({
                    'episode': episode,
                    'epsilon': model.epsilon,
                    'state_dict': model.state_dict(),
                    'is_best_model_by_far': True,
                    'time_step': best_time_step,
                }, is_best=True, filepath=checkpoint_folder_name + 'checkpoint-episode-%d.pth.tar' % episode)
                checkpoint_saved = True
                print('[main_processes.py] save the best checkpoint by far, episode={}, average time step={:.2f}'.format(
                    episode, avg_time_step))
        # case2: 保存检查点
        if episode % options.save_checkpoint_freq == 0 and not checkpoint_saved:
            if not model_evaluated:
                avg_time_step = evaluate_avg_time_step(model, episode)
                model_evaluated = True
            save_checkpoint({
                'episode': episode,
                'epsilon': model.epsilon,
                'state_dict': model.state_dict(),
                'is_best_model_by_far': False,
                'time_step': avg_time_step,
            }, is_best=False, filepath=checkpoint_folder_name + 'checkpoint-episode-%d.pth.tar' % episode)
            print('[main_processes.py] save a normal checkpoint, episode={}, average time step={:.2f}'.format(
                episode, avg_time_step))
            checkpoint_saved = True
        # case3: 不评估，继续下一个episode
        else:
            continue


def evaluate_avg_time_step(model, current_episode, test_episode_num=5):
    '''
    评估当前模型在数次游戏中坚持的平均时间，用于测试当前模型的游戏效果

    具体步骤：

    设n为测试游戏次数
    模型在与训练相同的gamestate下游玩n次，游玩过程采取的action全部为依据自身数据选择的，而非随机选取。n次游戏结束后，返回平均游戏时间。

    原作者设定n=5

    :param model: dqn model
    :param episode: current training episode
    :returns avg_time_step: 模型在n次游戏中坚持的平均时间
    '''
    model.set_eval()
    avg_time_step = 0.
    for test_case in range(test_episode_num):
        model.time_step = 0
        flappyBird = gamestate_for_training.GameState()
        o, r, terminal = flappyBird.frame_step([1, 0])
        o = preprocess(o)
        model.set_initial_state()
        while True:
            action = model.get_optim_action()
            o, r, terminal = flappyBird.frame_step(action)
            if terminal:
                break
            o = preprocess(o)
            model.current_state = np.append(
                model.current_state[1:, :, :], o.reshape((1,) + o.shape), axis=0)
            model.increase_time_step()
        avg_time_step += model.time_step
    avg_time_step /= test_episode_num
    print(
        '[main_processes.py] testing: episode: {}, average time step: {}'.format(
            current_episode,
            avg_time_step))
    return avg_time_step


def play_game_with_model(model_file_path, cuda=False, best=True):
    '''
    使用一个训练过的模型玩flappybird游戏

    :param weight: model file name containing weight of dqn
    :param best: if the model is best or not
    '''

    # 调试输出
    print('[main_processes.py] load pretrained model file: ' + model_file_path)
    model = network.FlappyBirdNetwork(epsilon=0., mem_size=0, cuda=cuda)
    load_checkpoint(model_file_path, model)

    model.set_eval()
    bird_game = gamestate_for_playing.GameState()
    model.set_initial_state()
    if cuda:
        model = model.cuda()
    while True:
        action = model.get_optim_action()
        o, r, terminal = bird_game.frame_step(action)
        if terminal:
            break
        o = preprocess(o)

        model.current_state = np.append(
            model.current_state[1:, :, :], o.reshape((1,) + o.shape), axis=0)

        model.increase_time_step()
    print('[main_processes.py] total time step is {}'.format(model.time_step))
